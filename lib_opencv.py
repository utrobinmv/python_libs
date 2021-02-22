import cv2
import numpy as np

def read_image_from_buffer(data):
    img_array = np.asarray(bytearray(data), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return image

def write_buffer_to_image(data):
    _, jpeg_frame = cv2.imencode('.jpg', data)

    str_data = jpeg_frame.tostring()

    byte_data = bytearray(jpeg_frame.tostring())

    return str_data

def is_this_an_image(image):
    try:
        width, height, color_channel_front = image.shape
    except:
        pass
    else:
        return True

    return False


def min_hw_image(image):
    try:
        width, height, color_channel_front = image.shape
    except:
        pass
    else:
        return min(width, height)

    return 0

def opencv_load_image_from_file(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    return image

def opencv_save_image_to_file(filename, image):
    cv2.imwrite(filename, image)

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def k_resize_image(image, min_size):
    # k_resize = 1
    (image_height, image_width, image_color) = image.shape

    # final_wide = 500
    final_wide = min_size #Выравнивание по меньшей стороне
    if image_height < image_width:
        # k_resize = image_height / final_wide
        r = float(final_wide) / image_height
    else:
        # k_resize = image_width / final_wide
        r = float(final_wide) / image_width

    return r

def resize_image(image, min_size, interpolation = cv2.INTER_AREA):

    #Изменим размеры фото
    (image_height, image_width, image_color) = image.shape

    r = k_resize_image(image, min_size)

    final_high = int(image_height * r)
    final_wide = int(image_width * r)
    dim = (final_wide, final_high)

    resized = cv2.resize(image, dim, interpolation = interpolation)

    return resized

def resize_image_wh(image, width, height, interpolation = cv2.INTER_AREA):
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = interpolation)
    return resized

def warp_images_card(im_src, poly, result_width, result_height):
    # Четыре точки соответствия на карте в исходном изображении

    x1,y1 = poly[0]
    x2,y2 = poly[1]
    x3,y3 = poly[2]
    x4,y4 = poly[3]

    pts_src = np.array([ [x1, y1], [x2, y2], [x3, y3],[x4, y4] ])
    # Прочитайте изображение для трансформации
    # im_dst = cv2.imread('/home/joefox/data/nextcloud/Projects/Unoto/Cards/tmp/cannyG.jpg')
    # Четыре точки соответствия на карте в изображении трансформации
    im_dst = np.zeros((result_height,result_width,3), np.uint8)
    
    pts_dst = np.array([ [0, 0],[result_width-1, 0],[result_width-1, result_height-1],[0, result_height-1] ])
    # Рассчитайте гомографию
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Трансформируем исходное изображение, используя полученную гомографию
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    return im_out

def cut_image_background(im_src_orig, box_rect):
    
    im_src = im_src_orig.copy()

    mask = np.zeros(im_src.shape[:2],np.uint8)

    (x1, y1, x2, y2) = box_rect

    rect = (x1,y1,x2-x1,y2-y1)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(im_src,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_cut = im_src*mask2[:,:,np.newaxis]

    return img_cut

def increase_brightness(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    
    new_v = np.int16(v)
    
    new_v[new_v > lim] = 255
    new_v[new_v <= lim] += value
    
    #Преобразуем обратно в uint8
    new_v[new_v < 0] = 0
    new_v[new_v > 255] = 255

    v = np.uint8(new_v)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def view_tesseract_boxes(img, boxes):
    '''
    Процедура отрисовывает прямоугольники сгенерированные функцией pytesseract.image_to_boxes
    '''
    h, w, c = img.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    

def image_cut_rectangle(image, x1, y1, x2, y2):
    return image[y1:y2,x1:x2]
