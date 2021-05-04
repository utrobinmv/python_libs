import tensorflow as tf
import numpy as np
import os

from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import visualization_utils

def enable_tf_gpu_dynamic_memory():
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)    

def read_pbtxt_to_dict(PATH_TO_LABELS):
    '''
    Функция возвращает словарь метох из файла brands.pbtxt
    где индекс - это метка, а name - значение
    
    Пример:
    PATH_TO_LABELS = "data_template_convert_pbtxt/brands.pbtxt"
    
    '''
    
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)    
    
    return category_index

def convert_pbtxt_dict_label(category_index):
    '''
    Функция конвертирует словарь id - name, в name - id
    где ключ это имя бренда, а число - id метка класса
    '''
    
    new_dict = {}
    for key in category_index.keys():
        new_dict[category_index[key]['name']] = category_index[key]['id']

    return new_dict

def image_one_to_tensor_vector(image_np, is_checkpoint):
    '''
    Функция конвертирует одну картинку numpy в вектор tf
    '''
    if is_checkpoint:
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    else:
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]    
    
    return input_tensor


def tf_load_model_from_savemodel(PATH_TO_SAVED_MODEL):
    '''
    Пример
    PATH_TO_SAVED_MODEL = "/home/joefox/data/nextcloud/Projects/Unoto/Cards_OLD_Max/model_tf/saved_model"
    '''
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    return detect_fn


def tf_load_model_from_checkpoint(PATH_TO_CFG_PIPELINE, PATH_TO_CKPT, name_checkpoint):
    '''
    Возвращает функцию, которую можно использовать как детектор изображения
    Пример:
     name_checkpoint = 'ckpt-70'
     input_tensor = image_one_to_tensor_vector(image_np)
     detect_fn = tf_load_model_from_checkpoint(PATH_TO_CFG_PIPELINE, PATH_TO_CKPT, name_checkpoint)
     detections, predictions_dict, shapes = detect_fn(input_tensor)
    '''
    
    @tf.function
    def detect_fn(image):
        """Detect objects in image."""
    
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
    
        #return detections, prediction_dict, tf.reshape(shapes, [-1])    
        return detections
    
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG_PIPELINE)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, name_checkpoint)).expect_partial()
    
    return detect_fn

def tf_visualization_boxes_and_labels_on_image(image_np_with_detections, is_checkpoint, detections, category_index, min_score_thresh, max_boxes_to_draw):
    if is_checkpoint:
        label_id_offset = 1
        
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False)
    else:    

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections
        
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False)
