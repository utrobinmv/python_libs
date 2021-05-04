import tensorflow as tf
from datasets import flowers
import pylab
 
import tf_slim as slim
 
DATA_DIR= "/home/joefox/Documents/tmp2"
 
# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', DATA_DIR)
 
# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
 
# Чтение данных в сеансе и отображение изображений с помощью Pylab
with tf.Session() as sess:
    # Инициализировать переменные
    sess.run(tf.global_variables_initializer())
    # Начать очередь
    tf.train.start_queue_runners()
    image_batch,label_batch = sess.run([image, label])
    # Показать картинки
    pylab.imshow(image_batch)
    pylab.show()
    
