import struct
import json
import numpy as np
from PIL import Image
import os
import io

def export_glb(filepath, vertices, faces, uvs=None, vcolors=None, textures=None, model_dir=None):
    # Group faces by unique texture
    face_groups = {}
    for i, face in enumerate(faces):
        tex = textures[i] if textures and i < len(textures) else None
        if tex not in face_groups:
            face_groups[tex] = []
        face_groups[tex].append(i)
    buffer_data = b''
    buffer_views = []
    accessors = []
    images = []
    samplers = []
    textures_json = []
    materials = []
    mesh_primitives = []
    tex_to_img_idx = {}
    tex_to_mat_idx = {}
    # Add all textures as images
    for tex in face_groups.keys():
        if tex is not None and tex not in tex_to_img_idx:
            tex_path = os.path.join(model_dir, tex) if model_dir else tex
            img = Image.open(tex_path)
            img = img.convert('RGBA')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_data = img_bytes.getvalue()
            images.append({'uri': os.path.basename(tex)})
            samplers.append({'magFilter': 9729, 'minFilter': 9729, 'wrapS': 10497, 'wrapT': 10497})
            textures_json.append({'sampler': len(samplers)-1, 'source': len(images)-1})
            tex_to_img_idx[tex] = len(images)-1
            tex_to_mat_idx[tex] = len(materials)
            materials.append({'pbrMetallicRoughness': {'baseColorTexture': {'index': len(textures_json)-1}}})
    if None in face_groups:
        tex_to_mat_idx[None] = len(materials)
        materials.append({'pbrMetallicRoughness': {}})
    # For each group, build local attribute sets and remap indices
    for tex, group in face_groups.items():
        local_verts = []
        local_uvs = []
        vert_map = {}
        indices = []
        for face_idx in group:
            face = faces[face_idx]
            face_uvs = uvs[face_idx] if uvs and uvs[face_idx] else [None]*len(face)
            for j, vi in enumerate(face):
                key = (vi, tuple(face_uvs[j]) if face_uvs[j] is not None else None)
                if key not in vert_map:
                    vert_map[key] = len(local_verts)
                    local_verts.append(vertices[vi])
                    if face_uvs[j] is not None:
                        local_uvs.append(face_uvs[j])
                indices.append(vert_map[key])
        # Write positions
        pos_offset = len(buffer_data)
        local_verts_np = np.array(local_verts, dtype=np.float32)
        buffer_data += local_verts_np.tobytes()
        buffer_views.append({'buffer': 0, 'byteOffset': pos_offset, 'byteLength': local_verts_np.nbytes, 'target': 34962})
        pos_accessor_idx = len(accessors)
        accessors.append({'bufferView': len(buffer_views)-1, 'byteOffset': 0, 'componentType': 5126, 'count': len(local_verts_np), 'type': 'VEC3',
            'min': local_verts_np.min(axis=0).tolist(), 'max': local_verts_np.max(axis=0).tolist()})
        # Write indices
        idx_offset = len(buffer_data)
        indices_np = np.array(indices, dtype=np.uint32)
        buffer_data += indices_np.tobytes()
        buffer_views.append({'buffer': 0, 'byteOffset': idx_offset, 'byteLength': indices_np.nbytes, 'target': 34963})
        idx_accessor_idx = len(accessors)
        accessors.append({'bufferView': len(buffer_views)-1, 'byteOffset': 0, 'componentType': 5125, 'count': len(indices_np), 'type': 'SCALAR',
            'min': [int(indices_np.min())], 'max': [int(indices_np.max())]})
        # Write UVs
        uv_accessor_idx = None
        if local_uvs:
            uv_offset = len(buffer_data)
            local_uvs_np = np.array(local_uvs, dtype=np.float32)
            buffer_data += local_uvs_np.tobytes()
            buffer_views.append({'buffer': 0, 'byteOffset': uv_offset, 'byteLength': local_uvs_np.nbytes, 'target': 34962})
            uv_accessor_idx = len(accessors)
            accessors.append({'bufferView': len(buffer_views)-1, 'byteOffset': 0, 'componentType': 5126, 'count': len(local_uvs_np), 'type': 'VEC2',
                'min': local_uvs_np.min(axis=0).tolist(), 'max': local_uvs_np.max(axis=0).tolist()})
        # Attributes
        mesh_attrs = {'POSITION': pos_accessor_idx}
        if uv_accessor_idx is not None:
            mesh_attrs['TEXCOORD_0'] = uv_accessor_idx
        mesh_primitives.append({
            'attributes': mesh_attrs,
            'indices': idx_accessor_idx,
            'material': tex_to_mat_idx[tex]
        })
    gltf = {
        'asset': {'version': '2.0'},
        'buffers': [{'byteLength': len(buffer_data)}],
        'bufferViews': buffer_views,
        'accessors': accessors,
        'images': images,
        'samplers': samplers,
        'textures': textures_json,
        'materials': materials,
        'meshes': [{
            'primitives': mesh_primitives
        }],
        'nodes': [{'mesh': 0}],
        'scenes': [{'nodes': [0]}],
        'scene': 0
    }
    while len(buffer_data) % 4 != 0:
        buffer_data += b'\x00'
    gltf_json = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(gltf_json) % 4 != 0:
        gltf_json += b' '
    glb = b''
    glb += struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(gltf_json) + 8 + len(buffer_data))
    glb += struct.pack('<I4s', len(gltf_json), b'JSON')
    glb += gltf_json
    glb += struct.pack('<I4s', len(buffer_data), b'BIN\x00')
    glb += buffer_data
    with open(filepath, 'wb') as f:
        f.write(glb) 