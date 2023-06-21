# lumos - 2023-05-21
# pixel#8843 @ discord
#
# 2023-05-20 -> 2023-05-21
#  - support for older versions of geo; -gv / --geoversion
#  - support for extracting textures to directory
#  - can now manually specify whether you want to use layouts; --nolayout

import argparse
import gltflib
import io
import numpy as np
import operator
import struct
import sys

from pathlib import Path
import PIL.Image

# A modified version of leamsii's DXT decompresser that supports:
#  - DXT3 
#  - passing bytes when decompressing
#  - correct orientation of images
# https://github.com/leamsii/Python-DXT-Decompress
class DXTBuffer:
    @staticmethod
    def __unpack(_bytes):
        STRUCT_SIGNS = {
        1 : 'B',
        2 : 'H',
        4 : 'I',
        8 : 'Q'
        }
        return struct.unpack('<' + STRUCT_SIGNS[len(_bytes)], _bytes)[0]

    @staticmethod
    def __unpackRGB(packed):
        r = (packed >> 11) & 0x1F
        g = (packed >> 5) & 0x3F
        b = (packed) & 0x1F

        r = (r << 3) | (r >> 2)
        g = (g << 2) | (g >> 4)
        b = (b << 3) | (b >> 2)

        return (r, g, b, 255)


    def __init__(self, w, h) -> None:
        self.w = w
        self.h = h

        self.block_countx = self.w // 4
        self.block_county = self.h // 4

        self.decompression_buffer = ['X'] * (self.w * self.h * 2)

    def __getColors(self, x, y, i, j, ctable, c0, c1, alpha):
        code = (ctable >> ( 2 * (4*j + i))) & 0x03
        pixel_color = None

        r0 = c0[0]
        g0 = c0[1]
        b0 = c0[2]

        r1 = c1[0]
        g1 = c1[1]
        b1 = c1[2]

        if code == 0:
            pixel_color = (r0, g0, b0, alpha)
        if code == 1:
            pixel_color = (r1, g1, b1, alpha)

        if c0 > c1:
            if code == 2:
                pixel_color = ((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3, alpha)
            if code == 3:
                pixel_color = ((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3, alpha)
        else:
            if code == 2:
                pixel_color = ((r0+r1)//2, (g0+g1)//2, (b0+b1)//2, alpha)
            if code == 3:
                pixel_color = (0, 0, 0, alpha)

        if (x + i) < self.w:
            self.decompression_buffer[(y + j) * self.w + (x + i)] = struct.pack('<B', pixel_color[0]) + struct.pack('<B', pixel_color[1]) + struct.pack('<B', pixel_color[2]) + struct.pack('<B', pixel_color[3])

    def decompressDXT1(self, buf):
        f = io.BytesIO(buf)
        for row in range(self.block_county):
            for col in range(self.block_countx):
                c0 = DXTBuffer.__unpack(f.read(2))
                c1 = DXTBuffer.__unpack(f.read(2))
                ctable = DXTBuffer.__unpack(f.read(4))

                for j in range(4):
                    for i in range(4):
                        self.__getColors(col * 4, row * 4, i, j, ctable, DXTBuffer.__unpackRGB(c0), DXTBuffer.__unpackRGB(c1), 255)
        
        return b''.join([_ for _ in self.decompression_buffer if _ != 'X'])

    def decompressDXT3(self, buf):
        f = io.BytesIO(buf)
        for row in range(self.block_county):
            for col in range(self.block_countx):
                atable = []
                for i in range(4):
                    alpha_data = DXTBuffer.__unpack(f.read(2))

                    atable.append(((alpha_data >> 0x0) & 0xf) * 17)
                    atable.append(((alpha_data >> 0x4) & 0xf) * 17)
                    atable.append(((alpha_data >> 0x8) & 0xf) * 17)
                    atable.append(((alpha_data >> 0xc) & 0xf) * 17)

                c0 = DXTBuffer.__unpack(f.read(2))
                c1 = DXTBuffer.__unpack(f.read(2))
                ctable = DXTBuffer.__unpack(f.read(4))

                for j in range(4):
                    for i in range(4):
                        self.__getColors(col * 4, row * 4, i, j, ctable, DXTBuffer.__unpackRGB(c0), DXTBuffer.__unpackRGB(c1), atable[j*4 + i])
        
        return b''.join([_ for _ in self.decompression_buffer if _ != 'X'])

class Unswizzler:
    def __copyBlockRect(self, sx, sy, w, h, dx, dy, src):
        PIXEL_SIZE = 4
        for y in range(h):
            for x in range(w):
                self.dst[((dy + y) * PIXEL_SIZE) * self.w + (dx + x) * PIXEL_SIZE + 0] = src[((sy + y) * PIXEL_SIZE) * self.w + (sx + x) * PIXEL_SIZE + 0]
                self.dst[((dy + y) * PIXEL_SIZE) * self.w + (dx + x) * PIXEL_SIZE + 1] = src[((sy + y) * PIXEL_SIZE) * self.w + (sx + x) * PIXEL_SIZE + 1]
                self.dst[((dy + y) * PIXEL_SIZE) * self.w + (dx + x) * PIXEL_SIZE + 2] = src[((sy + y) * PIXEL_SIZE) * self.w + (sx + x) * PIXEL_SIZE + 2]
                self.dst[((dy + y) * PIXEL_SIZE) * self.w + (dx + x) * PIXEL_SIZE + 3] = src[((sy + y) * PIXEL_SIZE) * self.w + (sx + x) * PIXEL_SIZE + 3]

    def toRGBA8888(self, w, h, data: bytes):
        self.w = w
        self.h = h
        self.buf = []

        self.src = []
        self.dst = []

        self.src[:] = data
        self.dst[:] = data
        
        stage = 0
        ts = 2
        while ts <= min(self.w, self.h):
            tmp = []
            tmp[:] = self.src
            self.src[:] = self.dst
            self.dst[:] = tmp

            sx = 0
            sy = 0
            for ty in range(0, self.h // ts):
                for tx in range(0, self.w // ts):
                    dx = tx * ts
                    dy = ty * ts

                    self.__copyBlockRect(sx, sy, ts // 2, ts // 2, dx, dy, self.src)
                    sx += ts // 2
                    if sx >= self.w:
                        sx = 0
                        sy += ts // 2

                    self.__copyBlockRect(sx, sy, ts // 2, ts // 2, dx + ts // 2, dy, self.src)
                    sx += ts // 2
                    if sx >= self.w:
                        sx = 0
                        sy += ts // 2
                    
                    self.__copyBlockRect(sx, sy, ts // 2, ts // 2, dx, dy + ts // 2, self.src)
                    sx += ts // 2
                    if sx >= self.w:
                        sx = 0
                        sy += ts // 2

                    self.__copyBlockRect(sx, sy, ts // 2, ts // 2, dx + ts // 2, dy + ts // 2, self.src)
                    sx += ts // 2
                    if sx >= self.w:
                        sx = 0
                        sy += ts // 2
            ts *= 2

        for y in range(h):
            for x in range(w):
                tmp = self.dst[(y * w * 4) + x * 4 + 0]
                self.dst[(y * w * 4) + x * 4 + 0] = self.dst[(y * w * 4) + x * 4 + 2]
                self.dst[(y * w * 4) + x * 4 + 2] = tmp

        return bytes(self.dst)

def unpack_i16(buf, offset):
    return struct.unpack_from('<h', buf, offset)[0]

def unpack_i32(buf, offset):
    return struct.unpack_from('<i', buf, offset)[0]

def unpack_r32(buf, offset):
    return offset + unpack_i32(buf, offset)

def unpack_str(buf, offset):
    str_offset = unpack_r32(buf, offset)
    chars = []
    while chr(buf[str_offset]) != '\x00':
        chars.append(chr(buf[str_offset]))
        str_offset += 1
    return ''.join(chars)

class Args:
    no_layout: bool = True
    flip_z: bool = True
    texture_output_directory: str = None
    geo_version: int = 2

class cGLTF(gltflib.GLTFModel):
    def __init__(self):
        self.accessors = []
        self.asset = gltflib.Asset(generator='generated by lumos - pixel#8843 @ discord')
        self.buffers = []
        self.bufferViews = []
        self.images = []
        self.materials = []
        self.nodes = []
        self.textures = []

        self.buffer = bytearray()
        self.resources = []
        self.stem = ''

    def __pad_buffer(self):
        self.buffer.extend(b'\xcc'*(16 - (len(self.buffer) % 16)))

    def __parse_mesh(self, bytes, offset, args: Args):
        name                    = unpack_str(bytes, offset + 0x04)
        flags                   = unpack_i16(bytes, offset + 0x0a)
        num_vertices            = unpack_i32(bytes, offset + 0x0c)
        ofs_vertices            = unpack_r32(bytes, offset + 0x10)
        num_indices             = unpack_i32(bytes, offset + 0x18)
        ofs_indices             = unpack_r32(bytes, offset + 0x1c)
        num_strips              = unpack_i32(bytes, offset + 0x20)
        ofs_strips              = unpack_r32(bytes, offset + 0x24)
        num_colormaps           = unpack_i32(bytes, offset + 0x74)
        ofs_colormaps           = unpack_r32(bytes, offset + 0xc4)
        num_blendshapes         = unpack_i32(bytes, offset + 0x6c)
        ofs_blendshapes         = unpack_r32(bytes, offset + 0x70)
        ofs_blendshape_data     = unpack_r32(bytes, offset + 0xc0)
        ofs_vertmappings        = unpack_r32(bytes, offset + 0xcc)
        num_blendshape_vertices = unpack_i32(bytes, offset + 0xd0)

        # glTF cannot handle above a certain amount of vertex info, let's limit
        # the 60+ colormaps that some objects can have.
        num_colormaps = min(num_colormaps, 32)

        self.__pad_buffer()

        # Each mesh name ends with '_X' where X is the index of that object in the
        # mesh table. This is the only reference to that index that I've found
        # contained in the mesh, but it seems really flimsy to expect each name to
        # end with it's number and there is probably another way the game
        # determines this. It works for now though.
        idx_mesh = int(name.split('_')[-1])

        # TODO:
        # If 'num_vertices' is 0 then that means the mesh is a collision mesh. They
        # are handled differently and I've figured out most of the data, but it's
        # not fully working so we'll set it to an empty mesh for now... with one
        # vertex to keep glTF validation happy since they don't like empty stuff.
        if num_vertices == 0:
            idx_bv_vertices = len(self.bufferViews)
            self.bufferViews.append(gltflib.BufferView(
                buffer=0,
                byteOffset=len(self.buffer),
                byteStride=12,
                byteLength=12,
                target=gltflib.BufferTarget.ARRAY_BUFFER.value
            ))

            self.buffer.extend(struct.pack('3f', *(0, 0, 0)))

            attributes = {
                'POSITION': len(self.accessors)
            }

            self.accessors.append(gltflib.Accessor( 
                byteOffset=0,
                count=1,
                bufferView=idx_bv_vertices,
                componentType=gltflib.ComponentType.FLOAT.value,
                type=gltflib.AccessorType.VEC3.value,
                min=(0, 0, 0),
                max=(0, 0, 0)
            ))

            self.meshes[idx_mesh] = gltflib.Mesh(
                name=name,
                primitives=[
                    gltflib.Primitive(
                        mode=gltflib.PrimitiveMode.POINTS.value,
                        attributes=attributes
                    )
                ]
            )
            return

        len_vertex = 40 if flags & 1 else 24

        len_gltf_vertex = 20 + (num_colormaps * 4)

        idx_bv_vertices = len(self.bufferViews)
        self.bufferViews.append(gltflib.BufferView(
            buffer=0,
            byteOffset=len(self.buffer),
            byteStride=len_gltf_vertex,
            byteLength=len_gltf_vertex*num_vertices,
            target=gltflib.BufferTarget.ARRAY_BUFFER.value
        ))

        colormaps = []
        for i in range(num_colormaps):
            colors = []
            ofs_colormap_header = ofs_colormaps + i * 0x10
            ofs_colors = unpack_r32(bytes, ofs_colormap_header + 0x00)
            for j in range(num_vertices):
                colors.append((
                    struct.unpack_from('<B', bytes, ofs_colors + i * 0x04 + 0x00)[0],
                    struct.unpack_from('<B', bytes, ofs_colors + i * 0x04 + 0x01)[0],
                    struct.unpack_from('<B', bytes, ofs_colors + i * 0x04 + 0x02)[0],
                    struct.unpack_from('<B', bytes, ofs_colors + i * 0x04 + 0x03)[0]
                ))

            colormaps.append(colors)

        v_positions = []
        for i in range(num_vertices):
            ofs_vertex = ofs_vertices + i * len_vertex

            position = struct.unpack_from('3f', bytes, ofs_vertex)
            v_positions.append(position)

            ofs_texcoord = 0x10 if len_vertex == 24 else 0x20
            texcoord = struct.unpack_from('2f', bytes, ofs_vertex + ofs_texcoord)

            self.buffer.extend(struct.pack('3f', *position))
            self.buffer.extend(struct.pack('2f', *texcoord))

            for idx_colormap in range(num_colormaps):
                self.buffer.extend(struct.pack('4B', *colormaps[idx_colormap][i]))

        attributes = {
            'POSITION':     len(self.accessors),
            'TEXCOORD_0':   len(self.accessors) + 1
        }

        self.accessors.append(gltflib.Accessor( 
            byteOffset=0,
            count=num_vertices,
            bufferView=idx_bv_vertices,
            componentType=gltflib.ComponentType.FLOAT.value,
            type=gltflib.AccessorType.VEC3.value,
            min=tuple(min(v_positions,key=operator.itemgetter(i))[i] for i in range(3)),
            max=tuple(max(v_positions,key=operator.itemgetter(i))[i] for i in range(3))
        ))

        self.accessors.append(gltflib.Accessor( 
            byteOffset=12,
            count=num_vertices,
            bufferView=idx_bv_vertices,
            componentType=gltflib.ComponentType.FLOAT.value,
            type=gltflib.AccessorType.VEC2.value
        ))

        for i in range(num_colormaps):
            attributes[f'COLOR_{i}'] = len(self.accessors)
            self.accessors.append(gltflib.Accessor(
                byteOffset=20 + (i * 4),
                count=num_vertices,
                bufferView=idx_bv_vertices,
                componentType=gltflib.ComponentType.UNSIGNED_BYTE.value,
                type=gltflib.AccessorType.VEC4.value,
                normalized=True
            ))

        morph_target_names = []
        morph_targets = []

        for i in range(num_blendshapes):
            position = [(0, 0, 0)] * num_vertices
            for j in range(num_vertices):
                idx_vertex = unpack_i32(bytes, ofs_vertmappings + j * 0x04)
                v_positions[j] = struct.unpack_from('3f', bytes, ofs_blendshape_data + (i * num_blendshape_vertices * 12) + (idx_vertex * 12))

            len_morph_vertex = 12

            idx_morph_vertices = len(self.bufferViews)
            self.bufferViews.append(gltflib.BufferView(
                buffer=0,
                byteOffset=len(self.buffer),
                byteStride=len_morph_vertex,
                byteLength=len_morph_vertex*num_vertices,
                target=gltflib.BufferTarget.ARRAY_BUFFER.value
            ))

            morph_attributes = {
                'POSITION': len(self.accessors)
            }

            self.accessors.append(gltflib.Accessor(
                byteOffset=0,
                count=num_vertices,
                bufferView=idx_morph_vertices,
                componentType=gltflib.ComponentType.FLOAT.value,
                type=gltflib.AccessorType.VEC3.value,
                min=tuple(min(v_positions,key=operator.itemgetter(i))[i] for i in range(3)),
                max=tuple(max(v_positions,key=operator.itemgetter(i))[i] for i in range(3))
            ))

            for j in range(num_vertices):
                self.buffer.extend(struct.pack('3f', *v_positions[j]))

            morph_target_names.append(unpack_str(bytes, ofs_blendshapes + i * 0x14))
            morph_targets.append(morph_attributes)

        idx_indices = 0
        primitives = []
        for i in range(num_strips):
            num_triangles = unpack_i32(bytes, ofs_strips + i * 16 + 0x00)
            idx_material = unpack_i32(bytes, ofs_strips + i * 16 + 0x04)

            # TODO: Figure out why idx_material sometimes is -1
            #if idx_material == -1:
            #    print(f'Tristrip {i} of mesh "{name}" has a texture index of -1!')
                
            indices = []
            for j in range(num_triangles):
                face = struct.unpack_from('HHH', bytes, ofs_indices + (idx_indices+j)*2)
                if len(set(face)) == len(face):
                    if j % 2 == 0:
                        indices.append(face[0])
                        indices.append(face[1])
                        indices.append(face[2])
                    else:
                        indices.append(face[0])
                        indices.append(face[2])
                        indices.append(face[1])

            self.__pad_buffer()
            idx_bufferview_indices = len(self.bufferViews)
            self.bufferViews.append(gltflib.BufferView(
                buffer=0,
                byteOffset=len(self.buffer),
                byteLength=len(indices)*2,
                target=gltflib.BufferTarget.ELEMENT_ARRAY_BUFFER.value
            ))

            for j in range(len(indices)):
                self.buffer.extend(struct.pack('H', indices[j]))

            idx_accessor_indices = len(self.accessors)
            self.accessors.append(gltflib.Accessor(
                count=len(indices),
                bufferView=idx_bufferview_indices,
                componentType=gltflib.ComponentType.UNSIGNED_SHORT.value,
                type=gltflib.AccessorType.SCALAR.value
            ))

            primitives.append(gltflib.Primitive(
                mode=gltflib.PrimitiveMode.TRIANGLES.value,
                indices=idx_accessor_indices,
                attributes=attributes,
                targets=morph_targets if len(morph_targets) > 0 else None,
                material=idx_material if idx_material >= 0 else None
            ))

            idx_indices += num_triangles + 2

        self.meshes[idx_mesh] = gltflib.Mesh(
            name=name,
            primitives=primitives,
            extras={
                'targetNames': morph_target_names
            } if len(morph_targets) > 0 else None
        )

    def __parse_texture(self, bytes, offset, args: Args):
        if args.geo_version == 0:
            w        = unpack_i32(bytes, offset + 0x04)
            h        = unpack_i32(bytes, offset + 0x08)
            format   = unpack_i32(bytes, offset + 0x0c)
            name     = unpack_str(bytes, offset + 0x14)
            len_data = unpack_i32(bytes, offset + 0x18)
            ofs_data = unpack_r32(bytes, offset + 0x34)
        elif args.geo_version == 1:
            w        = unpack_i32(bytes, offset + 0x04)
            h        = unpack_i32(bytes, offset + 0x08)
            format   = unpack_i32(bytes, offset + 0x0c)
            name     = unpack_str(bytes, offset + 0x14)
            len_data = unpack_i32(bytes, offset + 0x1c)
            ofs_data = unpack_r32(bytes, offset + 0x38)
        else:
            w        = unpack_i32(bytes, offset + 0x30)
            h        = unpack_i32(bytes, offset + 0x34)
            format   = unpack_i32(bytes, offset + 0x38)
            name     = unpack_str(bytes, offset + 0x40)
            len_data = unpack_i32(bytes, offset + 0x48)
            ofs_data = unpack_r32(bytes, offset + 0x64)

        if name == '':
            name = f'{self.stem}_{offset}'

        if not (format == 2 or format == 4 or format == 6):
            print(f'Unknown format value of {format} for texture \'{name}\'!')
            p_img = PIL.Image.new('RGB', (256, 256), (255, 0, 255))
        else:
            if format == 2:
                p_img = PIL.Image.frombytes('RGBA', (w, h), DXTBuffer(w, h).decompressDXT1(bytes[ofs_data:ofs_data+len_data]))
            elif format == 4:
                p_img = PIL.Image.frombytes('RGBA', (w, h), DXTBuffer(w, h).decompressDXT3(bytes[ofs_data:ofs_data+len_data]))
            elif format == 6:
                p_img = PIL.Image.frombytes('RGBA', (w, h), Unswizzler().toRGBA8888(w, h, bytes[ofs_data:ofs_data+len_data]))

        tmp = io.BytesIO()
        p_img.save(tmp, format='png')

        if args.texture_output_directory:
            p_img.save(f'{args.texture_output_directory}/{name}.png')

        self.materials.append(gltflib.Material(
            doubleSided=True,
            alphaMode=gltflib.AlphaMode.MASK.value,
            name=name,
            pbrMetallicRoughness=gltflib.PBRMetallicRoughness(
                baseColorTexture=gltflib.TextureInfo(
                    index=len(self.textures)
                )
            ),
            extensions={
                'KHR_materials_unlit': {}
            }
        ))

        self.textures.append(gltflib.Texture(
            source=len(self.images)
        ))

        self.images.append(gltflib.Image(
            name=name,
            bufferView=len(self.bufferViews),
            mimeType='image/png'
        ))

        self.bufferViews.append(gltflib.BufferView(
            buffer=len(self.buffers),
            byteLength=tmp.getbuffer().nbytes,
            byteOffset=len(self.buffer)
        ))

        self.buffer.extend(tmp.getvalue())

    @staticmethod
    def from_filepaths(geo_filepath: str, gdt_filepath: str, args: Args):
        with open(geo_filepath, 'rb') as f:
            geo_bytes = f.read()

        if gdt_filepath:
            with open(gdt_filepath, 'rb') as f:
                gdt_bytes = f.read()
            
        num_subfiles = unpack_i32(geo_bytes, 0x64)
        ofs_subfiles = unpack_r32(geo_bytes, 0x68)

        num_meshes = unpack_i32(geo_bytes, 0x0c)

        gltf = cGLTF()
        gltf.extensionsUsed = [
            'KHR_materials_unlit'
        ]

        gltf.stem = Path(geo_filepath).stem

        gltf.meshes = [None] * num_meshes

        geo_num_meshes = num_meshes
        geo_ofs_meshes = unpack_r32(geo_bytes, 0x10)

        if num_subfiles > 1 and gdt_filepath == None:
            print(f'GEO has subfiles, but accompanying GDT file has not been supplied!')
            sys.exit(1)

        for i in range(num_subfiles):
            ofs_subfile = ofs_subfiles + i * 0x4c

            gdt_subfile_offset = unpack_i32(geo_bytes, ofs_subfile + 0x1c)
            gdt_subfile_length = unpack_i32(geo_bytes, ofs_subfile + 0x20)

            if i != 0:
                gdt_subfile_header_num_meshes = unpack_i32(gdt_bytes, gdt_subfile_offset + 0x18)
                gdt_subfile_header_ofs_meshes = unpack_r32(gdt_bytes, gdt_subfile_offset + 0x1c)

                for j in range(gdt_subfile_header_num_meshes):
                    gdt_subfile_header_ofs_mesh = gdt_subfile_header_ofs_meshes + j * 0xd8
                    gltf.__parse_mesh(gdt_bytes, gdt_subfile_header_ofs_mesh, args)

                geo_num_meshes -= gdt_subfile_header_num_meshes

        for i in range(geo_num_meshes):
            ofs_mesh = geo_ofs_meshes + i * 0xd8
            gltf.__parse_mesh(geo_bytes, ofs_mesh, args)

        num_textures = unpack_i32(geo_bytes, 0x14)
        ofs_textures = unpack_r32(geo_bytes, 0x18)

        for i in range(num_textures):
            if args.geo_version == 0:
                len_texture = 0x38
            elif args.geo_version == 1:
                len_texture = 0x3c
            else:
                len_texture = 0x80

            ofs_texture = ofs_textures + i * len_texture
            gltf.__parse_texture(geo_bytes, ofs_texture, args)

        num_layouts = unpack_i32(geo_bytes, 0x2c)
        ofs_layouts = unpack_r32(geo_bytes, 0x30)

        if num_layouts > 0 and args.no_layout == False:
            print(f'{num_layouts} layouts found')
            num_objects = unpack_i32(geo_bytes, ofs_layouts + 0x64)
            ofs_objects = unpack_r32(geo_bytes, ofs_layouts + 0x68)

            print(f' {num_objects} objects found')

            for i in range(num_objects):
                x, y, z     = struct.unpack_from('fff', geo_bytes, ofs_objects + i * 0xa0 + 0x04)
                rx, ry, rz  = struct.unpack_from('fff', geo_bytes, ofs_objects + i * 0xa0 + 0x10)
                sx, sy, sz  = struct.unpack_from('fff', geo_bytes, ofs_objects + i * 0xa0 + 0x30)

                rx = -rx
                ry = -ry
                rz = -rz

                qx = np.sin(rx/2) * np.cos(ry/2) * np.cos(rz/2) - np.cos(rx/2) * np.sin(ry/2) * np.sin(rz/2)
                qy = np.cos(rx/2) * np.sin(ry/2) * np.cos(rz/2) + np.sin(rx/2) * np.cos(ry/2) * np.sin(rz/2)
                qz = np.cos(rx/2) * np.cos(ry/2) * np.sin(rz/2) - np.sin(rx/2) * np.sin(ry/2) * np.cos(rz/2)
                qw = np.cos(rx/2) * np.cos(ry/2) * np.cos(rz/2) + np.sin(rx/2) * np.sin(ry/2) * np.sin(rz/2)

                idx_mesh = unpack_i32(geo_bytes, ofs_objects + i * 0xa0 + 0x48)

                object_name = gltf.meshes[idx_mesh].name if idx_mesh >= 0 else f'{hex(ofs_objects + i * 0xa0)}'
                gltf.nodes.append(gltflib.Node(
                    name=object_name,
                    mesh=idx_mesh if idx_mesh >= 0 else None,
                    translation=(x, y, z),
                    rotation=(qx, qy, qz, qw),
                    scale=(sx, sy, sz)
                ))
        else:
            print(f'No layout, adding {len(gltf.meshes)} meshes to origin...')
            for i, mesh in enumerate(gltf.meshes):
                gltf.nodes.append(gltflib.Node(
                    name=mesh.name,
                    mesh=i
                ))

        gltf.nodes.append(gltflib.Node(
            name=gltf.stem,
            scale=(1, 1, -1) if args.flip_z else (1, 1, 1),
            children=[*range(0, len(gltf.nodes))]
        ))

        resource_name = f'{gltf.stem}.bin'
        gltf.buffers.append(gltflib.Buffer(
            byteLength=len(gltf.buffer),
            uri=resource_name
        ))

        gltf.resources.append(gltflib.FileResource(
            filename=resource_name,
            data=gltf.buffer
        ))

        return gltf              

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--geopath', type=str, required=True)
    parser.add_argument('-d', '--gdtpath', type=str, required=False)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-t', '--textureoutdir', type=str, required=False, help='specifies an output directory for textures')
    parser.add_argument('--noflip', action='store_true', help='disables inverting the z-axis')
    parser.add_argument('--nolayout', action='store_true', help='disables layout of all objects; may be needed for older geo versions')
    parser.add_argument('-gv', '--geoversion', type=int, default=2, help='specifies which version of geo should be used; default is 2')
    arguments = parser.parse_args()

    if arguments.geoversion < 0 or arguments.geoversion > 2:
        print(f'Invalid GEO version of {arguments.geoversion}! Only 0, 1, 2 are supported, with 2 being default.')
        sys.exit(1)

    args = Args()
    args.no_layout = arguments.nolayout
    args.flip_z = arguments.noflip == False
    args.texture_output_directory = arguments.textureoutdir
    args.geo_version = arguments.geoversion

    gltf = cGLTF.from_filepaths(arguments.geopath, arguments.gdtpath, args)
    gltflib.GLTF(
        gltf,
        gltf.resources
    ).export(arguments.output)
