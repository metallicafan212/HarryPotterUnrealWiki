# Written hastily by Pixel
# 2022-10-13

import functools
import itertools
import os
import shutil
import struct
import sys

from pathlib import Path

def read_i16(f):
    return struct.unpack('h', f.read(2))[0]

def read_i32(f):
    return struct.unpack('i', f.read(4))[0]

def read_f32(f):
    return struct.unpack('f', f.read(4))[0]

def read_ofs(f):
    return f.tell() + read_i32(f)

def read_str(f):
    offset = read_ofs(f)
    rw_ofs = f.tell()
    f.seek(offset)
    val = ''.join(iter(lambda: f.read(1).decode('ascii'), '\x00'))
    f.seek(rw_ofs)
    return val
    
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('geo2obj.py infile texture_directory output_directory')
        sys.exit(0)

    stem = Path(sys.argv[1]).stem

    f = open(sys.argv[1], 'rb')
    f.seek(12)

    num_meshes = read_i32(f)
    ofs_meshes = read_ofs(f)
    num_textures = read_i32(f)
    ofs_textures = read_ofs(f)

    mtl_buf = ''
    obj_buf = ''

    textures = []
    for i in range(num_textures):
        # A little hack for compatibility with older version of GEO files!
        if stem == 'cutscene':
            f.seek(ofs_textures + i * 56)
            
            f.seek(20, 1)
            textures.append(read_str(f))

        elif stem == 'oldharry' or stem == 'Burrow2':
            f.seek(ofs_textures + i * 60)

            f.seek(20, 1)
            textures.append(read_str(f))
        else:
            f.seek(ofs_textures + i * 128)

            f.seek(64, 1)
            textures.append(read_str(f))

        texture_subpath = None
        mtl_buf += f'newmtl {textures[len(textures)-1]}\n'
        if os.path.exists(sys.argv[2] + '/' + textures[len(textures)-1] + '.dds'):
            texture_subpath = textures[len(textures)-1] + '.dds'
        elif os.path.exists(sys.argv[2] + '/' + textures[len(textures)-1] + '.png'):
            texture_subpath = textures[len(textures)-1] + '.png'
        else:
            print(f'Couldn\'t find texture \"{textures[len(textures)-1]}\"')

        if texture_subpath is not None:
            shutil.copyfile(sys.argv[2] + '/' + texture_subpath, sys.argv[3] + '/' + texture_subpath)
            mtl_buf += f'map_Kd ' + texture_subpath + '\n'


    # Check for empty texture filenames to inform about stuff
    for s in textures:
        if s == '':
            print('!-------------!')
            print('Couldn\'t find some texture filenames.')
            print('This could be because the files is missing in the texture')
            print('directory.')
            print('It could also be because the GEO file is an older version')
            print('that doesn\'t have filenames. Fix for this will probably')
            print('be implemented in the full release.')
            print('')
            print('The mesh should still work, just without textures.')
            print('!-------------!')
            break

    obj_buf += 'mtllib ' + stem + '.mtl' + '\n'

    vertex_index = 0
    for i in range(num_meshes):
        f.seek(ofs_meshes + i * 216)

        f.seek(4, 1)
        name = read_str(f)
        f.seek(2, 1)
        bitflags = read_i16(f)
        num_vertices = read_i32(f)
        ofs_vertices = read_ofs(f)
        f.seek(4, 1)
        num_indices = read_i32(f)
        ofs_indices = read_ofs(f)
        num_strips = read_i32(f)
        ofs_strips = read_ofs(f)

        obj_buf += f'o {name}\n'

        vertex_size = 40 if bitflags & 1 else 24

        for j in range(num_vertices):
            f.seek(ofs_vertices + j * vertex_size)

            obj_buf += f'v {read_f32(f)} {read_f32(f)} {read_f32(f)}\n'
            if vertex_size == 24:
                f.seek(4, 1)
                obj_buf += f'vt {read_f32(f)} {1 - read_f32(f)}\n'
            else:
                f.seek(20, 1)
                obj_buf += f'vt {read_f32(f)} {1 - read_f32(f)}\n'

        indices = []
        for j in range(num_indices):
            f.seek(ofs_indices + j * 2)
            indices.append(read_i16(f))

        indices_index = 0
        for j in range(num_strips):
            f.seek(ofs_strips + j * 16)

            num_triangles = read_i32(f)
            texture_index = read_i32(f)
            
            if texture_index >= 0:
                obj_buf += f'usemtl {textures[texture_index]}\n'

            for k in range(num_triangles):
                v1 = -1
                v2 = -1
                v3 = -1

                if k & 1:
                    v1 = 1 + vertex_index + indices[indices_index + k]
                    v2 = 1 + vertex_index + indices[indices_index + k + 2]
                    v3 = 1 + vertex_index + indices[indices_index + k + 1]
                else:
                    v1 = 1 + vertex_index + indices[indices_index + k]
                    v2 = 1 + vertex_index + indices[indices_index + k + 1]
                    v3 = 1 + vertex_index + indices[indices_index + k + 2]

                obj_buf += f'f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n'
            
            indices_index = indices_index + num_triangles + 2

        vertex_index = vertex_index + num_vertices

    with open(sys.argv[3] + '/' + stem + '.mtl', 'w') as f:
        f.write(mtl_buf)

    with open(sys.argv[3] + '/' + stem + '.obj', 'w') as f:
        f.write(obj_buf)
