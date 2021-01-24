import json
import numpy as np
import math

from lib_pandas import create_dataframe, return_line_by_index
from lib_opencv import opencv_load_image_from_file, warp_images_card, resize_image_wh
from lib_geometry import max_angle_line, rotate
from lib_numpy import random_int

class CocoBoxes:
    list_boxes = ""
    def __init__(self, list_boxes):
        self.list_boxes = list_boxes
    def __len__(self):
        return len(self.list_boxes)
    def __getitem__(self, index):
        return CocoBox(return_line_by_index(self.list_boxes, index))

class CocoBox:
    category_name = ""
    name_box = ""
    image_filename = ""
    image_original = ""
    image_original_width = 0
    image_original_height = 0
    image_box = ""
    bbox = (0,0,0,0)
    def __init__(self, bbox_series):
        self.category_name = bbox_series['category_name']
        self.name_box = bbox_series['name_box']
        self.image_filename = bbox_series['image_filename']
        self.bbox = bbox_series['bbox']
    def load_image(self, folder_image):
        self.image_original = opencv_load_image_from_file(folder_image + self.image_filename)
        self.image_original_height, self.image_original_width, _ = self.image_original.shape
        

    def load_box(self):
        pass

    def load_box_resize(self, width_resize, height_resize, max_add_width, max_add_height, max_angle_size = 0, random_state = 0):
        '''
        Модифицирует необходимый бокс, под разными преобразованиями и углами наклона.
        '''
        
        def convert_point_to_rotate_int(point):
            x, y = point
            #return (int(y), int(x))
            return (int(x), int(y))
        
        def max_angle(angle, line):
            return max_angle_line(angle = rotate_angle, line = line) * 2
        
        
        x1, y1, width, height = self.bbox

        k1 = width / height
        k2 = width_resize / height_resize
        
        #x2 = x1 + width
        #y2 = y1 + height

        center_x = int(x1 + width / 2)
        center_y = int(y1 + height / 2)
       
        #Добавление с каждой стороны, делит значения на 2
        add_width = random_int(max_add_width // 2)
        add_height = random_int(max_add_height // 2)
        
        #Сдвиг текста влево в право
        k_leftright = np.random.random()
        
        #Сдвиг текста вверх вниз
        k_updown = np.random.random()
       
        #Определяем угол на который будем поворачивать
        rotate_angle = random_int(max_angle_size)
        
        #Случайным образом определим в какую сторону поворачивать
        rotate_party = random_int(1)
        if rotate_party == 1:
           rotate_angle = -rotate_angle
           
           
        
        if k1 <= k2:

            #Определяем высоту на которую необходимо будет поднять горизонтальные линии, для поворота заданного угла
            line = (x1, center_y, center_x, center_y)

            while True:
    
                rotate_lift_pix = max_angle(angle = rotate_angle, line = line)
                
                prom_add_height = max(add_height, rotate_lift_pix)
    
                #Выравниваем по Y
                new_height_resize = (height + prom_add_height * 2)
                k_resize = new_height_resize / height_resize
                new_width_resize = int(width_resize * k_resize)
        
                #min_height = (height + rotate_lift_pix * 2)
                #min_width = int(min_height * k1)
         
                if new_width_resize >= self.image_original_width:
                    if rotate_angle == 0:
                        break
                        
                    rotate_angle = rotate_angle / 2
                else:
                    break
            
            pass
        else:
            #Выравниваем по X
            new_width_resize = (width + add_width * 2)
            
            while True:
            
                line = (center_x - new_width_resize // 2, center_y, center_x, center_y)
                rotate_lift_pix = max_angle(angle = rotate_angle, line = line)
                
                prom_add_height = max(add_height, rotate_lift_pix)
                
                new_height_resize = (height + prom_add_height * 2)
                new_width_resize = int(new_height_resize * k1)
                               
                if new_width_resize >= self.image_original_width:
                    if rotate_angle == 0:
                        break
                        
                    rotate_angle = rotate_angle / 2
                else:
                    break
           
            line = (center_x - new_width_resize // 2, center_y, center_x, center_y)
            rotate_lift_pix = max_angle(angle = rotate_angle, line = line)
            
            #min_height = (height + rotate_lift_pix * 2)
            #min_width = new_width_resize
           
            
            #line = (center_x - new_width_resize // 2, center_y, center_x, center_y)
            #rotate_lift_pix = max_angle(angle = rotate_angle, line = line)
            
            #prom_add_height = max(add_height, rotate_lift_pix)

            #new_height_resize = (height + prom_add_height * 2)
            #new_width_resize = int(new_height_resize * k1)
            
            ##Выравниваем по X
            #new_width_resize = (width + add_width * 2)
            #k_resize = new_width_resize / width
            #new_height_resize = int(height * k_resize)
            
            pass


        #Получаем размер сдвига по width
        add_width = new_width_resize // 2 - width // 2
        
        min_part = width // 2
        delta_part = new_width_resize // 2 - min_part
        
        max_left = center_x - min_part
        max_right = self.image_original_width - center_x - min_part
        
        #Добавим ограничение min_part относительно рамок фото
        min_space = (self.image_original_width - new_width_resize) // 2
        
        
        b = delta_part - min_space
        if b > 0:            
            if b < min(max_left, max_right):
                add_part = b
                min_part += add_part
                max_left -= add_part
                max_right -= add_part
                pass
        
        if max_left < max_right:
            left_party = int(min(add_width, max_left) * k_leftright) + min_part
            right_party = new_width_resize - left_party
        else:
            right_party = int(min(add_width, max_right) * k_leftright) + min_part
            left_party = new_width_resize - right_party
            
        
        #left_party = int(left_shift * k_leftright) + min_part

        #Если выходит за рамки считаем с другой стороны
        #if center_x - left_party < 0 or center_x + right_party > self.image_original_width:
            ##left_party = ((max_left - min_part) * k_leftright) + min_part
            
            #right_party = int((max_right - min_part) * k_leftright) + min_part
            #left_party = new_width_resize - right_party
        
        #Получаем размер сдвига по height
        min_up = height // 2
        up_shift = new_height_resize // 2 - min_up
        up_party = int(up_shift * k_updown) + min_up
        down_party = new_height_resize - up_party
        
        origin = (center_x, center_y)

        #Левая верхняя точка прямоугольника
        point = (center_x - left_party, center_y - up_party)
        point_left_up = rotate(origin, point, math.radians(rotate_angle))
        
        #Правая верхняя точка прямоугольника
        point = (center_x + right_party, center_y - up_party)
        point_right_up = rotate(origin, point, math.radians(rotate_angle))
        
        #Левая нижняя точка прямоугольника
        point = (center_x - left_party, center_y + down_party)
        point_left_down = rotate(origin, point, math.radians(rotate_angle))
        
        #Правая нижняя точка прямоугольника
        point = (center_x + right_party, center_y + down_party)
        point_right_down = rotate(origin, point, math.radians(rotate_angle))

        poly = []
        poly.append(convert_point_to_rotate_int(point_left_up))
        poly.append(convert_point_to_rotate_int(point_right_up))
        poly.append(convert_point_to_rotate_int(point_right_down))
        poly.append(convert_point_to_rotate_int(point_left_down))

        img_out = warp_images_card(self.image_original, poly, new_width_resize, new_height_resize)
        
        img_text = resize_image_wh(img_out, width_resize, height_resize)
        
        self.image_box = img_text

def coco_load_from_file(file_coco_annotation):
    with open(file_coco_annotation) as json_data:
        data_dict = json.load(json_data)
        json_data.close()

    return data_dict

def coco_list_categories(data_dict):
    data_categories = data_dict['categories']

    list_categories = []

    for data in data_categories:
        list_categories.append(data['name'])

    return list_categories

def coco_read_categorie_bbox(data_dict, categorie_name):

    data_images = data_dict['images']
    data_annotations = data_dict['annotations']
    data_categories = data_dict['categories']

    images = []
    images_id = []

    categories = []
    categories_id = []
    categories_bbox = []

    for image in data_images:
        images.append((image['id'], image['file_name'], image['width'], image['height']))
        images_id.append(image['id'])

    premium_categorie_id = []

    for categorie in data_categories:

        cn = categorie['name']
        categorie_id = categorie['id']

        if cn == categorie_name:
            premium_categorie_id.append(categorie_id)

        categories.append((categorie_id, cn))
        categories_id.append(categorie_id)

    bbox_width_all = []
    bbox_height_all = []

    for annotation in data_annotations:
        image_bbox = annotation['bbox']
        category_id = annotation['category_id']
        image_id = annotation['image_id']

        metadata = annotation['metadata']

        name_box = metadata['name']

        x1 = int(image_bbox[0])
        y1 = int(image_bbox[1])
        width = int(image_bbox[2])
        height = int(image_bbox[3])

        try:
            index_premium = premium_categorie_id.index(category_id)
        except:
            continue
        else:
            bbox_width_all.append(width)
            bbox_height_all.append(height)

        image_idx = images_id.index(image_id)
        image_id, image_filename, image_width, image_height = images[image_idx]

        category_idx = categories_id.index(category_id)
        category_id, category_name = categories[category_idx]

        bbox = (x1, y1, width, height)
        categories_bbox.append((category_name, name_box, image_filename, bbox))

    df = create_dataframe(categories_bbox, ['category_name', 'name_box', 'image_filename', 'bbox'])
    
    return df
