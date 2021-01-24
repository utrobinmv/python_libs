import math
import numpy as np

def get_intersect(lineA, lineB):
    '''
    #Функция возвращает точку пересечения двух прямых
    '''

    (x1_1,y1_1,x2_1,y2_1) = lineA
    (x1_2,y1_2,x2_2,y2_2) = lineB

    a1 = (x1_1,y1_1)
    a2 = (x2_1,y2_1)
    b1 = (x1_2,y1_2)
    b2 = (x2_2,y2_2)

    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        # return (float('inf'), float('inf'))
        return (-1, -1)
    return (x/z, y/z)

def distance(x1, y1, x2, y2):
    '''
    # Функция возвращает растояние между двумя точками
    '''

    c = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return c

def point_on_piece(line, point):
    '''
    # Функция возвращает лежит ли заданная точка на заданном отрезке прмямой
    '''

    x1, y1, x2, y2 = line
    x, y = point

    res = (x-x1) * (y2-y1) - (x2-x1) * (y-y1)

    # print("res: " + str(res))

    if res == 0:
        #Площадь фигуры равна нулю, необходимо определить, что лежит на отрезке.
        if x <= max(x1, x2) and x >= min(x1, x2) and y <= max(y1, y2) and y >= min(y1, y2):
            return True

    return False

def point_on_line(line, point):
    '''
    # Функция возвращает растояние точки до прямой
    '''

    x1, y1, x2, y2 = line
    x, y = point

    dx1 = x2 - x1
    dy1 = y2 - y1

    dx = x - x1
    dy = y - y1

    s = dx1 * dy - dx * dy1 #Считаем удвоенную (ориентированной) площадь треугольника, если она равна 0, то все точки лежат на одной линии

    ab = math.sqrt(dx1 * dx1 + dy1 * dy1)

    h = abs(s / ab)

    # d = 5

    # print(f"line (x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}) and point (x:{x}, y:{y}) = h:{h}")

    return h

def angle_line(line):
    '''
    #Функция возвращает угол направления прямой относительно оси координат
    '''

    angle = 0

    (x1,y1,x2,y2) = line
    angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

    return angle


def max_angle_line(angle, line):
    '''
    Работает не оптимально
    Функция ищет, количество точек на которое максимально можно поднять линию, чтобы получить заданный угол
    '''
    (x1_start,y1_start,x2_start,y2_start) = line
    
    start_angle = angle_line(line)
    
    abs_angle = abs(angle)
    
    if start_angle > abs_angle:
        return 0

    num_y = 0

    while True:
        num_y += 1
        line = (x1_start,y1_start,x2_start,y2_start + num_y)
        start_angle = angle_line(line)
        if start_angle > abs_angle:
            return num_y - 1
    
    return 0
    
    
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    
    Поверните точку против часовой стрелки на заданный угол вокруг заданного начала координат.

    Угол следует задавать в радианах.
    
    math.radians(10) - преобразование градусов в радианы
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    

    return qx, qy