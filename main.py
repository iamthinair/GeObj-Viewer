import sys
import os
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QMessageBox
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective
from PIL import Image
import numpy as np
import geobj_loader
import glb_exporter

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

    def load_model(self, vertices, faces, uvs=None, textures=None, materials=None, model_dir=None, vcolors=None, face_vcolor_indices=None):
        self.model_vertices = vertices
        self.model_faces = faces
        self.model_uvs = uvs
        self.model_textures = textures
        self.model_materials = materials
        self.texture_cache = {}
        self.model_dir = model_dir
        self.model_vcolors = vcolors
        self.model_face_vcolor_indices = face_vcolor_indices
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

    def load_texture(self, filename: Optional[str]) -> Optional[int]:
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
        gluPerspective(45.0, aspect, 0.01, 1000.0)
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
            if self.model_textures:
                tex_file = self.model_textures[i] if isinstance(self.model_textures[i], str) else None
                if tex_file:
                    tex_id = self.load_texture(tex_file)
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
                    glTexCoord2f(u, 1.0 - v)
                glVertex3fv(self.model_vertices[idx])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING)
        glDisable(GL_LIGHT0)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_mode = 'rotate'
        elif event.button() == Qt.RightButton:
            self.mouse_mode = 'pan'
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        if self.mouse_mode == 'rotate':
            self.camera_azimuth -= dx * 0.5
            self.camera_elevation += dy * 0.5
            self.camera_elevation = max(-89, min(89, self.camera_elevation))
        elif self.mouse_mode == 'pan':
            self.camera_pan_x += dx * 0.01 * self.camera_distance
            self.camera_pan_y -= dy * 0.01 * self.camera_distance
        self.last_mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.mouse_mode = None
        self.last_mouse_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.camera_distance *= 0.9 ** delta
        self.camera_distance = max(self.camera_distance_min, min(self.camera_distance_max, self.camera_distance))
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeObj Viewer')
        self.setGeometry(100, 100, 800, 600)
        self.glWidget = GLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.init_menu()

    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        export_action = QAction('Export as GLB', self)
        export_action.triggered.connect(self.export_glb_dialog)
        file_menu.addAction(export_action)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open GeObj File', '', 'GeObj Files (*.obj);;All Files (*.*)')
        if file_path:
            model_dir = os.path.dirname(file_path)
            try:
                geobj_data = geobj_loader.load_geobj(file_path)
                self.glWidget.load_model(
                    geobj_data['vertices'],
                    geobj_data['faces'],
                    uvs=geobj_data['face_uvs'],
                    textures=geobj_data['textures'],
                    materials=geobj_data['materials'],
                    model_dir=geobj_data['model_dir'],
                    vcolors=geobj_data['vcolors'],
                    face_vcolor_indices=geobj_data['face_vcolor_indices']
                )
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