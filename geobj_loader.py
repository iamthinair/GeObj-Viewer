import os
import re

def load_geobj(filepath):
    # Only allow files exported by Banjo's Backpack
    with open(filepath, 'r') as f:
        first_lines = [f.readline().rstrip('\n\r') for _ in range(10)]
    found_banjo = any(line == "# Exported with Banjo's Backpack" for line in first_lines)
    if not found_banjo:
        raise Exception('This loader only supports GeObj files exported by Banjo\'s Backpack.')
    vertices = []
    vcolors = []
    texcoords = []
    faces = []
    face_uvs = []
    face_vcolor_indices = []
    materials = []
    textures = []
    current_material = None
    model_dir = os.path.dirname(filepath)
    mtl_file = None
    mtl_data = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    vcolor_pattern = re.compile(r'#vcolor (\d+) (\d+) (\d+) (\d+)')
    fvcolorindex_pattern = re.compile(r'#fvcolorindex (.+)')
    for i, line in enumerate(lines):
        if line.startswith('mtllib'):
            mtl_file = line.split()[1].strip()
        elif line.startswith('v '):
            parts = line.strip().split()
            vertices.append(tuple(float(x) for x in parts[1:4]))
        elif line.startswith('#vcolor'):
            match = vcolor_pattern.match(line.strip())
            if match:
                color = tuple(int(match.group(j))/255.0 for j in range(1,5))
                vcolors.append(color)
        elif line.startswith('vt '):
            parts = line.strip().split()
            texcoords.append(tuple(float(x) for x in parts[1:3]))
        elif line.startswith('usemtl'):
            current_material = line.split()[1].strip()
        elif line.startswith('f '):
            face = []
            uv = []
            for v in line.strip().split()[1:]:
                if '/' in v:
                    vparts = v.split('/')
                    vi = int(vparts[0]) - 1
                    ti = int(vparts[1]) - 1 if len(vparts) > 1 and vparts[1] else None
                else:
                    vi = int(v) - 1
                    ti = None
                face.append(vi)
                uv.append(texcoords[ti] if ti is not None and ti < len(texcoords) else None)
            faces.append(face)
            face_uvs.append(uv)
            materials.append(current_material)
        elif line.startswith('#fvcolorindex'):
            match = fvcolorindex_pattern.match(line.strip())
            if match:
                indices = [int(idx)-1 for idx in match.group(1).split()]
                face_vcolor_indices.append(indices)
            else:
                face_vcolor_indices.append(None)
    # Parse MTL file for texture info and material properties
    if mtl_file:
        mtl_path = os.path.join(model_dir, mtl_file)
        if os.path.exists(mtl_path):
            with open(mtl_path, 'r') as mtl:
                cur_mat = None
                for line in mtl:
                    if line.startswith('newmtl'):
                        cur_mat = line.split()[1].strip()
                        mtl_data[cur_mat] = {}
                    elif cur_mat:
                        tokens = line.strip().split()
                        if not tokens:
                            continue
                        if tokens[0] == 'Ka':
                            mtl_data[cur_mat]['Ka'] = [float(x) for x in tokens[1:4]]
                        elif tokens[0] == 'Kd':
                            mtl_data[cur_mat]['Kd'] = [float(x) for x in tokens[1:4]]
                        elif tokens[0] == 'Ks':
                            mtl_data[cur_mat]['Ks'] = [float(x) for x in tokens[1:4]]
                        elif tokens[0] == 'Ns':
                            mtl_data[cur_mat]['Ns'] = float(tokens[1])
                        elif tokens[0] == 'd':
                            mtl_data[cur_mat]['d'] = float(tokens[1])
                        elif tokens[0] == 'Tr':
                            mtl_data[cur_mat]['d'] = 1.0 - float(tokens[1])
                        elif tokens[0] == 'illum':
                            mtl_data[cur_mat]['illum'] = int(tokens[1])
                        elif tokens[0] == 'map_Kd':
                            mtl_data[cur_mat]['map_Kd'] = tokens[1]
                        elif tokens[0] == 'map_Ka':
                            mtl_data[cur_mat]['map_Ka'] = tokens[1]
                        elif tokens[0] == 'map_Ks':
                            mtl_data[cur_mat]['map_Ks'] = tokens[1]
                        elif tokens[0] == 'map_Bump' or tokens[0] == 'bump':
                            mtl_data[cur_mat]['map_Bump'] = tokens[1]
    face_materials = []
    textures = []
    for mat in materials:
        mat_props = mtl_data.get(mat, {}) if mat else {}
        face_materials.append(mat_props)
        if 'map_Kd' in mat_props:
            textures.append(mat_props['map_Kd'])
        elif 'texture' in mat_props:
            textures.append(mat_props['texture'])
        else:
            textures.append(None)
    while len(face_vcolor_indices) < len(faces):
        face_vcolor_indices.append(None)
    return {
        'vertices': vertices,
        'vcolors': vcolors,
        'faces': faces,
        'face_uvs': face_uvs,
        'face_vcolor_indices': face_vcolor_indices,
        'materials': face_materials,
        'textures': textures,
        'model_dir': model_dir
    } 