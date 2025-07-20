import sys
import os
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton
from PyQt5.QtWidgets import QDoubleSpinBox, QLabel, QHBoxLayout
from PyQt5.QtGui import QPalette, QColor, QFont, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective
from PIL import Image
import numpy as np
import geobj_loader
import glb_exporter
from PyQt5.QtWidgets import QComboBox

class DevWindow(QDialog):
    def __init__(self, glwidget, mainwindow, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Dev Tools')
        self.setMinimumSize(350, 400)
        self._layout = QVBoxLayout()
        self.vflip_checkbox = QCheckBox('Flip V (Texture UVs)')
        self.vflip_checkbox.setChecked(glwidget.v_flip)
        self.vflip_checkbox.stateChanged.connect(lambda state: glwidget.set_v_flip(state == Qt.Checked))
        self._layout.addWidget(self.vflip_checkbox)
        self.reset_cam_btn = QPushButton('Reset Camera')
        self.reset_cam_btn.clicked.connect(glwidget.reset_camera)
        self._layout.addWidget(self.reset_cam_btn)
        self._layout.addWidget(QLabel('Level Scale'))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(4)
        self.scale_spin.setRange(0.0001, 1000.0)
        self.scale_spin.setSingleStep(0.01)
        self.scale_spin.setValue(mainwindow.level_scale)
        self.scale_spin.valueChanged.connect(mainwindow.set_level_scale)
        self._layout.addWidget(self.scale_spin)
        self.setLayout(self._layout)

class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_vertices: Optional[List[Tuple[float, float, float]]] = None
        self.model_faces: Optional[List[List[int]]] = None
        self.model_uvs: Optional[List[List[Optional[Tuple[float, float]]]]] = None
        self.model_textures: Optional[List[Optional[str]]] = None
        self.model_materials: Optional[List[dict]] = None
        self.texture_cache = {}
        self.model_dir: Optional[str] = None
        self.camera_distance = 8.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0
        self.last_mouse_pos = None
        self.mouse_mode = None
        self.model_offset = [0.0, 0.0, 0.0]
        self.camera_distance_min = 1.0
        self.camera_distance_max = 100.0
        self.model_vcolors: Optional[List[Tuple[float, float, float, float]]] = None
        self.model_face_vcolor_indices: Optional[List[Optional[List[int]]]] = None
        self.v_flip = False  # Default to False (disabled)
        self.show_skeleton = False
        self.skeleton_joints = []
        self.skeleton_connections = []
        self.texture_clamp_settings = {}  # tex_path -> str ('repeat', 'clamp', 'mirror')
        self.uv_clamp = False
        self.uv_white_oob = False
        self.camera_near = 0.1
        self.camera_far = 1000.0

    def load_model(self, vertices, faces, uvs=None, textures=None, materials=None, model_dir=None, vcolors=None, face_vcolor_indices=None, joints=None, connections=None, scale=1.0):
        # Apply scale to vertices and joints
        if scale != 1.0:
            vertices = [(x*scale, y*scale, z*scale) for (x, y, z) in vertices]
            if joints is not None:
                joints = [(x*scale, y*scale, z*scale) for (x, y, z) in joints]
        self.model_vertices = vertices
        self.model_faces = faces
        self.model_uvs = uvs
        self.model_textures = textures
        self.model_materials = materials
        self.texture_cache = {}
        self.model_dir = model_dir
        # Populate texture_clamp_settings with all unique texture paths
        self.texture_clamp_settings = {}
        if textures:
            for tex in textures:
                if tex:
                    if self.model_dir and not os.path.isabs(tex):
                        tex_path = os.path.normpath(os.path.join(self.model_dir, tex))
                    else:
                        tex_path = tex
                    self.texture_clamp_settings[tex_path] = self.texture_clamp_settings.get(tex_path, 'repeat')
        self.model_vcolors = vcolors
        self.model_face_vcolor_indices = face_vcolor_indices
        self.skeleton_joints = joints if joints is not None else []
        self.skeleton_connections = connections if connections is not None else []
        self.reset_model_pos()
        self.auto_fit_camera()
        self.update()

    def reset_model_pos(self):
        if self.model_vertices:
            min_v = [min(v[i] for v in self.model_vertices) for i in range(3)]
            max_v = [max(v[i] for v in self.model_vertices) for i in range(3)]
            center = [(min_v[i] + max_v[i]) / 2 for i in range(3)]
            self.model_offset = [float(-c) for c in center]
        else:
            self.model_offset = [0.0, 0.0, 0.0]
        self.update()

    def auto_fit_camera(self):
        if not self.model_vertices or len(self.model_vertices) == 0:
            return
        min_v = np.min(np.array(self.model_vertices), axis=0)
        max_v = np.max(np.array(self.model_vertices), axis=0)
        center = (min_v + max_v) / 2
        size = np.linalg.norm(max_v - min_v)
        fov = 45.0
        aspect = self.width() / self.height() if self.height() != 0 else 1
        distance = size / (2 * np.tan(np.radians(fov / 2)))
        self.camera_distance = distance * 1.2
        self.camera_distance_min = max(distance * 0.2, 0.01)
        self.camera_distance_max = distance * 5.0
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0
        self.camera_azimuth = 45.0
        self.camera_elevation = 30.0
        self.model_offset = (-center).tolist()
        # Robust near/far plane setup
        self.camera_near = max(0.1, size / 500.0)
        self.camera_far = size * 10.0

    def load_texture(self, filename: Optional[str], material_name: Optional[str] = None) -> Optional[int]:
        if not filename:
            return None
        if self.model_dir and not os.path.isabs(filename):
            tex_path = os.path.normpath(os.path.join(self.model_dir, filename))
        else:
            tex_path = filename
        if tex_path in self.texture_cache:
            return self.texture_cache[tex_path]
        try:
            img = Image.open(tex_path)
            img = img.convert('RGBA')
            img_data = img.tobytes('raw', 'RGBA', 0, -1)
            width, height = img.size
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            # Use user-selected clamp setting if available
            mode = self.texture_clamp_settings.get(tex_path, 'repeat')
            if mode == 'clamp':
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            elif mode == 'mirror':
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            self.texture_cache[tex_path] = tex_id
            return tex_id
        except Exception as e:
            print(f'Failed to load texture {tex_path}: {e}')
            self.texture_cache[tex_path] = None
            return None

    def set_material(self, mat: Optional[dict]):
        if not mat:
            return
        ka = mat.get('Ka', [0.2, 0.2, 0.2])
        kd = mat.get('Kd', [0.8, 0.8, 0.8])
        ks = mat.get('Ks', [0.0, 0.0, 0.0])
        ns = mat.get('Ns', 0.0)
        d = mat.get('d', 1.0)
        glColor4f(*kd, d)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ka + [d])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, kd + [d])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, ks + [d])
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, ns)

    def initializeGL(self):
        glClearColor(0.13, 0.13, 0.13, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h != 0 else 1
        # Use robust near/far planes
        gluPerspective(45.0, aspect, self.camera_near, self.camera_far)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        az, el = np.radians(self.camera_azimuth), np.radians(self.camera_elevation)
        cx = self.camera_distance * np.cos(el) * np.sin(az)
        cy = self.camera_distance * np.sin(el)
        cz = self.camera_distance * np.cos(el) * np.cos(az)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glTranslatef(self.camera_pan_x, self.camera_pan_y, 0)
        gluLookAt(cx, cy, cz, 0, 0, 0, 0, 1, 0)
        glTranslatef(*self.model_offset)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        for i, face in enumerate(self.model_faces or []):
            mat = self.model_materials[i] if self.model_materials else None
            self.set_material(mat)
            tex_id = None
            material_name = None
            if mat and isinstance(mat, dict) and 'name' in mat:
                material_name = mat['name']
            elif self.model_materials and isinstance(self.model_materials, list):
                if hasattr(self, 'model_material_names') and self.model_material_names:
                    material_name = self.model_material_names[i]
            if self.model_textures:
                tex_file = self.model_textures[i] if isinstance(self.model_textures[i], str) else None
                if tex_file:
                    tex_id = self.load_texture(tex_file, material_name)
            if tex_id:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex_id)
            else:
                glDisable(GL_TEXTURE_2D)
            uvs = self.model_uvs[i] if self.model_uvs else None
            vcolor_indices = self.model_face_vcolor_indices[i] if self.model_face_vcolor_indices else None
            if len(face) == 3:
                glBegin(GL_TRIANGLES)
            elif len(face) == 4:
                glBegin(GL_QUADS)
            else:
                glBegin(GL_POLYGON)
            for j, idx in enumerate(face):
                # Set vertex color if available, otherwise white
                if vcolor_indices and self.model_vcolors and j < len(vcolor_indices) and vcolor_indices[j] < len(self.model_vcolors):
                    r, g, b, a = self.model_vcolors[vcolor_indices[j]]
                    glColor4f(r, g, b, a)
                else:
                    glColor4f(1, 1, 1, 1)
                # Set texture coordinate if available
                if tex_id and uvs and uvs[j] is not None:
                    u, v = uvs[j]
                    if self.v_flip:
                        v = 1.0 - v
                    oob = (u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0)
                    if self.uv_white_oob and oob:
                        glColor4f(1, 1, 1, 1)
                        # Do not call glTexCoord2f, just set white color
                    else:
                        if self.uv_clamp:
                            u = min(max(u, 0.0), 1.0)
                            v = min(max(v, 0.0), 1.0)
                        glTexCoord2f(u, v)
                glVertex3fv(self.model_vertices[idx])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)
        # Draw skeleton if enabled
        if self.show_skeleton and self.skeleton_joints and self.skeleton_connections:
            glDisable(GL_DEPTH_TEST)
            glColor3f(1, 1, 0)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            for a, b in self.skeleton_connections:
                if a < len(self.skeleton_joints) and b < len(self.skeleton_joints):
                    glVertex3fv(self.skeleton_joints[a])
                    glVertex3fv(self.skeleton_joints[b])
            glEnd()
            glPointSize(6.0)
            glBegin(GL_POINTS)
            for j in self.skeleton_joints:
                glVertex3fv(j)
            glEnd()
            glEnable(GL_DEPTH_TEST)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_mode = 'rotate'
        elif event.button() == Qt.RightButton:
            self.mouse_mode = 'pan'
        self.last_mouse_pos = event.pos()

    def reset_camera(self):
        self.auto_fit_camera()
        self.update()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        # Scale pan speed with camera distance and model size
        pan_scale = 0.01 * self.camera_distance
        if self.mouse_mode == 'rotate':
            self.camera_azimuth -= dx * 0.5
            self.camera_elevation += dy * 0.5
            self.camera_elevation = max(-89, min(89, self.camera_elevation))
        elif self.mouse_mode == 'pan':
            self.camera_pan_x += dx * pan_scale
            self.camera_pan_y -= dy * pan_scale
        self.last_mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.mouse_mode = None
        self.last_mouse_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        # Zoom speed scales with distance
        zoom_factor = 0.9 ** delta
        self.camera_distance *= zoom_factor
        self.camera_distance = max(self.camera_distance_min, min(self.camera_distance_max, self.camera_distance))
        self.update()

    def set_v_flip(self, flip: bool):
        self.v_flip = flip
        self.update()
    def set_show_skeleton(self, show: bool):
        self.show_skeleton = show
        self.update()

    def set_texture_clamp(self, tex_path, mode):
        self.texture_clamp_settings[tex_path] = mode
        # Remove from cache so it reloads with new clamp mode
        if tex_path in self.texture_cache:
            del self.texture_cache[tex_path]
        self.update()

    def set_uv_clamp(self, clamp: bool):
        self.uv_clamp = clamp
        self.update()

    def set_uv_white_oob(self, enable: bool):
        self.uv_white_oob = enable
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeObj .obj File Viewer')
        self.setWindowIcon(QIcon('jiggy.ico'))
        self.setGeometry(100, 100, 800, 600)
        self.glWidget = GLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.init_menu()
        self.dev_window = None
        self.init_shortcuts()
        self.level_scale = 1.0
    def set_level_scale(self, value):
        self.level_scale = value
        # Reload the current model at new scale if loaded
        if hasattr(self, 'last_geobj_data') and self.last_geobj_data:
            self.load_geobj_data(self.last_geobj_data)
    def load_geobj_data(self, geobj_data):
        self.glWidget.load_model(
            geobj_data['vertices'],
            geobj_data['faces'],
            uvs=geobj_data['face_uvs'],
            textures=geobj_data['textures'],
            materials=geobj_data['materials'],
            model_dir=geobj_data['model_dir'],
            vcolors=geobj_data['vcolors'],
            face_vcolor_indices=geobj_data['face_vcolor_indices'],
            joints=geobj_data.get('joints'),
            connections=geobj_data.get('connections'),
            scale=self.level_scale
        )
    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        export_action = QAction('Export as GLB', self)
        export_action.triggered.connect(self.export_glb_dialog)
        file_menu.addAction(export_action)
        
        # Add Edit menu
        edit_menu = menubar.addMenu('Edit')
        self.skeleton_action = QAction('Show Skeleton', self)
        self.skeleton_action.setCheckable(True)
        self.skeleton_action.setChecked(self.glWidget.show_skeleton)
        self.skeleton_action.triggered.connect(lambda checked: self.glWidget.set_show_skeleton(checked))
        edit_menu.addAction(self.skeleton_action)
    def init_shortcuts(self):
        dev_action = QAction(self)
        dev_action.setShortcut('Ctrl+D')
        dev_action.triggered.connect(self.open_dev_window)
        self.addAction(dev_action)
        reset_cam_action = QAction(self)
        reset_cam_action.setShortcut('R')
        reset_cam_action.triggered.connect(self.glWidget.reset_camera)
        self.addAction(reset_cam_action)
    def open_dev_window(self):
        if self.dev_window is None:
            self.dev_window = DevWindow(self.glWidget, self, self)
        self.dev_window.scale_spin.setValue(self.level_scale)
        self.dev_window.show()
        self.dev_window.raise_()
        self.dev_window.activateWindow()
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open GeObj File', '', 'GeObj/OBJ Files (*.obj);;All Files (*)')
        if file_path:
            model_dir = os.path.dirname(file_path)
            try:
                geobj_data = geobj_loader.load_geobj(file_path)
                self.last_geobj_data = geobj_data
                self.load_geobj_data(geobj_data)
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))

    def export_glb_dialog(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Export as GLB', '', 'glTF Binary (*.glb)')
        if file_path:
            gw = self.glWidget
            glb_exporter.export_glb(
                file_path,
                gw.model_vertices,
                gw.model_faces,
                uvs=gw.model_uvs,
                vcolors=getattr(gw, 'model_vcolors', None),
                textures=gw.model_textures,
                model_dir=gw.model_dir
            )

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('jiggy.ico'))
    app.setStyle('Fusion')
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(60, 120, 200))
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(dark_palette)
    app.setFont(QFont('Segoe UI', 10))
    window = MainWindow()
    window.setStyleSheet('background-color: #222; color: #fff;')
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 
