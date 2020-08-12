# Maria Isabel Ortiz Naranjo
# Graficas por computadora
# SR4 - Flat shading
# 18176
# Basado en codigo proporcionado en clase por Dennis 

import struct
from Obj import Obj
import random
from collections import namedtuple

def char(c):
    return struct.pack('=c', c.encode('ascii'))

def word(c):
    return struct.pack('=h', c)

def dword(c):
    return struct.pack('=l', c)

def color(r, g, b):
    return bytes([b, g, r])
def normalizeColorArray(colors_array):
    return [round(i*255) for i in colors_array]

V2 = namedtuple('Vertex2', ['x', 'y'])
V3 = namedtuple('Vertex3', ['x', 'y', 'z'])

def sum(v0, v1):
   
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
   
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
   
    return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
   
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def length(v0):
    
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):
   
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)


def bbox(*vertices):
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]

    xs.sort()
    ys.sort()

    xmin = xs[0]
    xmax = xs[-1]
    ymin = ys[0]
    ymax = ys[-1]

    return xmin, xmax, ymin, ymax

def cross(v1, v2):
    return V3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x,
    )

def barycentric(A, B, C, P):
    cx, cy, cz = cross(
        V3(B.x - A.x, C.x - A.x, A.x - P.x),
        V3(B.y - A.y, C.y - A.y, A.y - P.y),
    )

    if abs(cz) < 1:
        return -1, -1, -1

    u = cx/cz
    v = cy/cz
    w = 1 - (cx + cy) / cz

    return w, v, u


NEGRO = color(0,0,0)
BLANCO = color(1,1,1)


class Render(object):

    def __init__(self):
        self.pixels = []
        self.width = 800
        self.height = 800
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = 800
        self.viewport_height = 800
        self.glClear()

        self.zbuffer = [
            [-9999999 for x in range(self.width)]
            for y in range(self.height)
        ]

    def glCreateWindow(self,width, height):
        self.width = width
        self.height = height
        self.glClear()
        self.glViewport(0, 0, width, height)


    def point(self, x, y, color):
        self.pixels[y][x] = color

    def glViewport(self, x, y, width, height):
        self.vpX = x
        self.vpY = y
        self.vpWidth = width
        self.vpHeight = height

    def glClear(self):
        BLACK = color(0,0,0)
        self.pixels = [
            [BLACK for x in range(self.width)] for y in range(self.height)
        ]

    def glClearColor(self, r=1, g=1, b=1):
        normalized = normalizeColorArray([r,g,b])
        clearColor = color(normalized[0], normalized[1], normalized[2])

        self.pixels = [
            [clearColor for x in range(self.width)] for y in range(self.height)
        ]
    
    def glVertex(self, x, y):
        pixelX = ( x + 1) * (self.vpWidth  / 2 ) + self.vpX
        pixelY = ( y + 1) * (self.vpHeight / 2 ) + self.vpY

        try:
            self.pixels[round(pixelY)][round(pixelX)] = self.curr_color
        except:
            pass

    def glVertex_coord(self, x, y):
        try:
            self.pixels[y][x] = self.curr_color
        except:
            pass
    
    def glCoordinate(self, value, is_vertical):
        real_coordinate = ((value+1) * (self.viewport_height/2) + self.viewport_y) if is_vertical else ((value+1) * (self.viewport_width/2) + self.viewport_x)
        return round(real_coordinate)

    def glColor(self, r, g, b):
        normalized = normalizeColorArray([r,g,b])
        self.color = color(normalized[0], normalized[1], normalized[2])

    def triangle1(self, A, B, C, color=None):
        if A.y > B.y:
            A, B = B, A
        if A.y > C.y:
            A, C = C, A
        if B.y > C.y:
            B, C = C, B

        dx_ac = C.x - A.x
        dy_ac = C.y - A.y

        if dy_ac == 0:
            return

        mi_ac = dx_ac/dy_ac

        dx_ab = B.x - A.x
        dy_ab = B.y - A.y

        if dy_ab != 0:
            mi_ab = dx_ab/dy_ab

            for y in range(A.y, B.y + 1):
                xi = round(A.x - mi_ac * (A.y - y))
                xf = round(A.x - mi_ab * (A.y - y))

                if xi > xf:
                    xi, xf = xf, xi
                for x in range(xi, xf + 1):
                    self.point(x, y, color)

        dx_bc = C.x - B.x
        dy_bc = C.y - B.y

        if dy_bc:

            mi_bc = dx_bc/dy_bc

            for y in range(B.y, C.y + 1):
                xi = round(A.x - mi_ac * (A.y - y))
                xf = round(B.x - mi_bc * (B.y - y))

                if xi > xf:
                    xi, xf = xf, xi
                for x in range(xi, xf + 1):
                    self.point(x, y, color)

    def glLine(self, x0, y0, x1, y1): # NDC
        #x0 = round(( x0 + 1) * (self.vpWidth  / 2 ) + self.vpX)
        #x1 = round(( x1 + 1) * (self.vpWidth  / 2 ) + self.vpX)
        #y0 = round(( y0 + 1) * (self.vpHeight / 2 ) + self.vpY)
        #y1 = round(( y1 + 1) * (self.vpHeight / 2 ) + self.vpY)

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        offset = 0
        limit = 0.5
        
        y = y0

        for x in range(x0, x1 + 1):
            if steep:
                self.glVertex_coord(y, x)
            else:
                self.glVertex_coord(x, y)

            offset += 2*dy
            if offset >= dx:
                y += -1 if y0 < y1 else -1
                limit += 2*dx

    def glLine_coord(self, x0, y0, x1, y1): # Window coordinates

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        offset = 0
        limit = 0.5
        
        try:
            m = dy/dx
        except ZeroDivisionError:
            pass
        else:
            y = y0

            for x in range(x0, x1 + 1):
                if steep:
                    self.glVertex_coord(y, x)
                else:
                    self.glVertex_coord(x, y)

                offset += m
                if offset >= limit:
                    y += 1 if y0 < y1 else -1
                    limit += 1

    def loadModel(self, filename, translate=[0,0], scale=[1,1]):
        model = Obj(filename)

        light = V3(0, 0, 1)

        for face in model.faces:
            vcount = len(face)

            if vcount == 3:
                face1 = face[0][0] - 1
                face2 = face[1][0] - 1
                face3 = face[2][0] - 1

                v1 = model.vertices[face1]
                v2 = model.vertices[face2]
                v3 = model.vertices[face3]

                x1 = round((v1[0] * scale[0]) + translate[0])
                y1 = round((v1[1] * scale[1]) + translate[1])
                z1 = round((v1[2] * scale[2]) + translate[2])

                x2 = round((v2[0] * scale[0]) + translate[0])
                y2 = round((v2[1] * scale[1]) + translate[1])
                z2 = round((v2[2] * scale[2]) + translate[2])

                x3 = round((v3[0] * scale[0]) + translate[0])
                y3 = round((v3[1] * scale[1]) + translate[1])
                z3 = round((v3[2] * scale[2]) + translate[2])

                a = V3(x1, y1, z1)
                b = V3(x2, y2, z2)
                c = V3(x3, y3, z3)

                normal = cross(sub(b, a), sub(c, a))
                intensity = dot(norm(normal), norm(light))
                grey = round(255 * intensity)
                if grey < 0:
                    continue

                intensity_color = color(grey, grey, grey)
                self.triangle(a, b, c, intensity_color)

            else:
                face1 = face[0][0] - 1
                face2 = face[1][0] - 1
                face3 = face[2][0] - 1
                face4 = face[3][0] - 1

                v1 = model.vertices[face1]
                v2 = model.vertices[face2]
                v3 = model.vertices[face3]
                v4 = model.vertices[face4]

                x1 = round((v1[0] * scale[0]) + translate[0])
                y1 = round((v1[1] * scale[1]) + translate[1])
                z1 = round((v1[2] * scale[2]) + translate[2])

                x2 = round((v2[0] * scale[0]) + translate[0])
                y2 = round((v2[1] * scale[1]) + translate[1])
                z2 = round((v2[2] * scale[2]) + translate[2])

                x3 = round((v3[0] * scale[0]) + translate[0])
                y3 = round((v3[1] * scale[1]) + translate[1])
                z3 = round((v3[2] * scale[2]) + translate[2])

                x4 = round((v4[0] * scale[0]) + translate[0])
                y4 = round((v4[1] * scale[1]) + translate[1])
                z4 = round((v4[2] * scale[2]) + translate[2])

                a = V3(x1, y1, z1)
                b = V3(x2, y2, z2)
                c = V3(x3, y3, z3)
                d = V3(x4, y4, z4)

                normal = cross(sub(b, a), sub(c, a))

                intensity = dot(norm(normal), norm(light))
                grey = round(255 * intensity)
                if grey < 0:
                    continue

                intensity_color = color(grey, grey, grey)

                self.triangle(a, b, c, intensity_color)
                self.triangle(a, c, d, intensity_color)
    
    def triangle(self, A, B, C, color):
        xmin, xmax, ymin, ymax = bbox(A, B, C)

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                P = V2(x, y)
                w, v, u = barycentric(A, B, C, P)
                if w < 0 or v < 0 or u < 0:
                    continue

                z = A.z * w + B.z * v + C.z * u

                if z > self.zbuffer[x][y]:
                    self.point(x, y, color)
                    self.zbuffer[x][y] = z

    def glFinish(self, filename='output.bmp'):
        f = open(filename, 'wb')
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        # image header
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        for x in range(self.height):
            for y in range(self.width):
                f.write(self.pixels[x][y])

        f.close()

r = Render()
# r.loadModel('./models/chicken.obj', (378,189), (378,378) )
r.loadModel('./models/chicken.obj', translate=(700, 300, 200), scale = (300, 300, 350) )
r.glFinish('output.bmp')
