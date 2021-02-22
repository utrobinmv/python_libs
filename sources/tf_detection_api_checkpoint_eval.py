import time
import random
import tensorflow as tf

import cv2
import os

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

import numpy as np

def resize_image_wh(image, width, height, interpolation = cv2.INTER_AREA):
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = interpolation)
    return resized

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def convert_dict_label(category_index):
   new_dict = {}
   for key in category_index.keys():
      new_dict[category_index[key]['name']] = category_index[key]['id']
         
   return new_dict

#GRAPH_PB_PATH = '/home/joefox/data/nextcloud/Projects/Unoto/Cards_OLD_Max/model_tf/saved_model.pb'
#with tf.compat.v1.Session() as sess:
   #print("load graph")
   #with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
       #graph_def = tf.compat.v1.GraphDef()
   #graph_def.ParseFromString(f.read())
   #sess.graph.as_default()
   #tf.import_graph_def(graph_def, name='')
   #graph_nodes=[n for n in graph_def.node]
   #names = []
   #for t in graph_nodes:
      #names.append(t.name)
   #print(names)


PATH_TO_LABELS = "/home/joefox/data/nextcloud/Projects/Unoto/Cards_OLD_Max/data_template_convert_pbtxt/brands.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

dict_category = convert_dict_label(category_index)


load_path = '/home/joefox/data/nextcloud/Projects/Unoto/Cards_OLD_Max/data_template_test_convert_imgs/'
#load_path = '/home/joefox/Documents/orig2/'
#save_path = '/home/joefox/Documents/recog/'
#save_path = '/home/joefox/Documents/recog2/'

stop_label = 10
view_time = False

max_x = 480
max_y = 480

label_id_offset = 1

list_images = []
#list_labels = []

#vector = []

list_dir = os.listdir(load_path)
for idx, file in enumerate(list_dir):

   #print('Running inference for {}... '.format(image_path), end='')

   filename = load_path + file

   image_np = cv2.imread(filename, cv2.IMREAD_COLOR)
   
   image_np = resize_image_wh(image_np, max_x, max_y)

   brand_split = file.split('-0')
   brand_name = brand_split[0]
   
   category_id = dict_category[brand_name]
   
   input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

   list_images.append((input_tensor, category_id))

   # if idx == 0:
   #    vector = np.expand_dims(image_np, 0)
   # else:
   #    vector = np.concatenate([vector, np.expand_dims(image_np, 0)]) 
   
   # list_images.append(image_np)
   #list_labels.append(category_id)

   # if idx == 6:
   #    break

random.shuffle(list_images)

print('Длина списка: ',len(list_images), ', stop label:', stop_label)

a = 1
# input_tensor = tf.convert_to_tensor(vector, dtype=tf.float32)

PATH_TO_MODEL_DIR = "/home/joefox/Downloads/my_centernet_resnet101_v1_fpn_512x512_coco17_tpu-8"

PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

list_ckpt = []

list_dir = os.listdir(PATH_TO_CKPT,)
for file in list_dir:
   if file.endswith(".index"):
      lst_split = file.split('.')[0].split('ckpt-')
      checkpoint_num = lst_split[1]
      checkpoint_num = int(checkpoint_num)
      
      # print(checkpoint_num)
      
      list_ckpt.append(checkpoint_num)
   
list_ckpt.sort()
   
a = 0

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

list_ckpt = list_ckpt[40:]
   
for now_ckpt in list_ckpt:
      
   checkpoint_num = now_ckpt
   #checkpoint_num = list_ckpt[24]
   
   if view_time:
      print('Loading model...', checkpoint_num, end='')
   start_time = time.time()
   
   # Restore checkpoint
   ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
   ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-' + str(checkpoint_num))).expect_partial()
   #ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-' + str(checkpoint_num))).assert_existing_objects_matched()

   @tf.function
   def detect_fn(image):
      """Detect objects in image."""

      image, shapes = detection_model.preprocess(image)
      prediction_dict = detection_model.predict(image, shapes)
      detections = detection_model.postprocess(prediction_dict, shapes)

      return detections, prediction_dict, tf.reshape(shapes, [-1])
    
   end_time = time.time()
   elapsed_time = end_time - start_time
   if view_time:
      print(' ...Done! Took {} seconds'.format(elapsed_time))
   
   all_answer = len(list_images)
   correct_answer = 0
   uncorrect_answer = 0

   # detections = detect_fn(input_tensor)

   # end_time = time.time()
   # elapsed_time = end_time - start_time
   # print('Detect! Took {} seconds'.format(elapsed_time))

   # num_detections = int(detections.pop('num_detections'))
   # detections = {key: value[0, :num_detections].numpy()
   #                for key, value in detections.items()}
   # detections['num_detections'] = num_detections
   
   # # detection_classes should be ints.
   # detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

   if stop_label > 0:
      all_answer = stop_label

   a = 1

   for idx, el_list in enumerate(list_images):

      input_tensor, label = el_list

      # input_tensor = np.expand_dims(image_np, 0)
      #detections = detect_fn(input_tensor)
      detections, predictions_dict, shapes = detect_fn(input_tensor)
   
      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(detections.pop('num_detections'))
      detections = {key: value[0, :num_detections].numpy()
                     for key, value in detections.items()}
      detections['num_detections'] = num_detections
      
      # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
      

      if len(detections['detection_classes']) > 0:
         if detections['detection_classes'][0] + label_id_offset == label:
            correct_answer += 1
         else:
            uncorrect_answer += 1
      else:
         uncorrect_answer += 1

      if view_time:
         print('.', end='')
         
      a = 1

      if stop_label > 0 and stop_label == idx:
         break
      
   score = correct_answer / all_answer

   if view_time:
      print('')
   print('checkpoint_num:',checkpoint_num, ', score:', score)

   end_time = time.time()
   elapsed_time = end_time - start_time
   if view_time:
      print('Detect! Took {} seconds'.format(elapsed_time))


   #image_np_with_detections = image_np.copy()
   
   #viz_utils.visualize_boxes_and_labels_on_image_array(
       #image_np_with_detections,
          #detections['detection_boxes'],
          #detections['detection_classes'],
          #detections['detection_scores'],
          #category_index,
          #use_normalized_coordinates=True,
          #max_boxes_to_draw=1,
          #min_score_thresh=.5,
          #agnostic_mode=False)
   
   
   #save_file = save_path + file
   #cv2.imwrite(save_path + file, image_np_with_detections)
   #print(save_file)
   
   #plt.figure()
   #plt.imshow(image_np_with_detections)

print('Done')
