import numpy as np
import cv2

import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D, Activation
from keras.layers import Input, Dense, Concatenate

def rle_decode(mask_rle, shape=(1280, 1918, 1)):
    '''
    Функция расшифровывает маску изображения, особый архив масок mask_rle
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    img = img.reshape(shape)
    return img

def keras_generator(gen_df, batch_size):
    '''
    Функция генерирует батч из картинок
    Пример использования
    for x, y in keras_generator(train_df, 16):
        break
    '''
    
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):
            img_name, mask_rle = gen_df.sample(1).values[0]
            img = cv2.imread('data/train/{}'.format(img_name))
            mask = rle_decode(mask_rle)
            
            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            
            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255.
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)
        
        
def load_optim(num):
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    return adam
        
def load_model(num):
    
    if num == 1:
        base_model = ResNet50(weights='imagenet', input_shape=(256,256,3), include_top=False)
    elif num == 2:
        base_model = VGG16(weights='imagenet', input_shape=(256,256,3), include_top=False)
     
    base_out = base_model.output #Выходной тензор модели 
    
    return base_model
    
def base_model():
    '''
    Функция возвращает модель на основе VGG16 формирующая вероятности к классу, используется для задач сегментации
    '''
    
    base_model = VGG16(weights='imagenet', input_shape=(256,256,3), include_top=False)
    
    
    base_out = base_model.output
    
    
    up = UpSampling2D(32, interpolation='bilinear')(base_out)
    conv = Conv2D(1, (1, 1))(up)
    conv = Activation('sigmoid')(conv)
    
    model = Model(input=base_model.input, output=conv)    
    
    return model


def keras_learn_model(model, batch_x, train_df, val_df):
    '''
    Функция обучает модель
    
    
    Пример использования
    df = pd.read_csv('data/train_masks.csv')
    df.shape
    train_df = df[:4000]
    val_df = df[4000:]
    for batch_x, y in keras_generator(train_df, 16):
        break
    model = base_model()
    
    '''
    
    
    
    best_w = keras.callbacks.ModelCheckpoint('fcn_best.h5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='auto',
                                    period=1)
    
    last_w = keras.callbacks.ModelCheckpoint('fcn_last.h5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=True,
                                    mode='auto',
                                    period=1)
    
    
    callbacks = [best_w, last_w]
    
    
    
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    
    model.compile(adam, 'binary_crossentropy')    
    
    batch_size = 16
    model.fit_generator(keras_generator(train_df, batch_size),
                  steps_per_epoch=100,
                  epochs=3,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=keras_generator(val_df, batch_size),
                  validation_steps=50,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0)    
    
    
    pred = model.predict(batch_x)
    
    '''
    Вывод результата картинки и маски
    im_id = 13
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
    axes[0].imshow(x[im_id])
    axes[1].imshow(pred[im_id, ..., 0] > 0.5)
    
    plt.show()
    '''
    
    return pred
        
def create_model_segmentation(num):
    
    if num == 1:
        
        inp = Input(shape=(256, 256, 3))
        
        conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
        conv_1_1 = Activation('relu')(conv_1_1)
        
        conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
        conv_1_2 = Activation('relu')(conv_1_2)
        
        pool_1 = MaxPooling2D(2)(conv_1_2)
        
        
        conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
        conv_2_1 = Activation('relu')(conv_2_1)
        
        conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
        conv_2_2 = Activation('relu')(conv_2_2)
        
        pool_2 = MaxPooling2D(2)(conv_2_2)
        
        
        conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
        conv_3_1 = Activation('relu')(conv_3_1)
        
        conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
        conv_3_2 = Activation('relu')(conv_3_2)
        
        pool_3 = MaxPooling2D(2)(conv_3_2)
        
        
        conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
        conv_4_1 = Activation('relu')(conv_4_1)
        
        conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
        conv_4_2 = Activation('relu')(conv_4_2)
        
        pool_4 = MaxPooling2D(2)(conv_4_2)
        
        up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4)
        conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(up_1)
        conv_up_1_1 = Activation('relu')(conv_up_1_1)
        
        conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
        conv_up_1_2 = Activation('relu')(conv_up_1_2)
        
        
        up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
        conv_up_2_1 = Conv2D(128, (3, 3), padding='same')(up_2)
        conv_up_2_1 = Activation('relu')(conv_up_2_1)
        
        conv_up_2_2 = Conv2D(128, (3, 3), padding='same')(conv_up_2_1)
        conv_up_2_2 = Activation('relu')(conv_up_2_2)
        
        
        up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
        conv_up_3_1 = Conv2D(64, (3, 3), padding='same')(up_3)
        conv_up_3_1 = Activation('relu')(conv_up_3_1)
        
        conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
        conv_up_3_2 = Activation('relu')(conv_up_3_2)
        
        
        
        up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
        conv_up_4_1 = Conv2D(32, (3, 3), padding='same')(up_4)
        conv_up_4_1 = Activation('relu')(conv_up_4_1)
        
        conv_up_4_2 = Conv2D(1, (3, 3), padding='same')(conv_up_4_1)
        result = Activation('sigmoid')(conv_up_4_2)
        
        
        model = Model(inputs=inp, outputs=result)
        
        
    elif num == 2: #model Unet
        inp = Input(shape=(256, 256, 3))
        
        conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
        conv_1_1 = Activation('relu')(conv_1_1)
        
        conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
        conv_1_2 = Activation('relu')(conv_1_2)
        
        pool_1 = MaxPooling2D(2)(conv_1_2)
        
        
        conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
        conv_2_1 = Activation('relu')(conv_2_1)
        
        conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
        conv_2_2 = Activation('relu')(conv_2_2)
        
        pool_2 = MaxPooling2D(2)(conv_2_2)
        
        
        conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
        conv_3_1 = Activation('relu')(conv_3_1)
        
        conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
        conv_3_2 = Activation('relu')(conv_3_2)
        
        pool_3 = MaxPooling2D(2)(conv_3_2)
        
        
        conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
        conv_4_1 = Activation('relu')(conv_4_1)
        
        conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
        conv_4_2 = Activation('relu')(conv_4_2)
        
        pool_4 = MaxPooling2D(2)(conv_4_2)
        
        up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4)
        conc_1 = Concatenate()([conv_4_2, up_1])
        
        conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(conc_1)
        conv_up_1_1 = Activation('relu')(conv_up_1_1)
        
        conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
        conv_up_1_2 = Activation('relu')(conv_up_1_2)
        
        
        up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_2)
        conc_2 = Concatenate()([conv_3_2, up_2])
        
        conv_up_2_1 = Conv2D(128, (3, 3), padding='same')(conc_2)
        conv_up_2_1 = Activation('relu')(conv_up_2_1)
        
        conv_up_2_2 = Conv2D(128, (3, 3), padding='same')(conv_up_2_1)
        conv_up_2_2 = Activation('relu')(conv_up_2_2)
        
        
        up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_2)
        conc_3 = Concatenate()([conv_2_2, up_3])
        
        conv_up_3_1 = Conv2D(64, (3, 3), padding='same')(conc_3)
        conv_up_3_1 = Activation('relu')(conv_up_3_1)
        
        conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
        conv_up_3_2 = Activation('relu')(conv_up_3_2)
        
        
        
        up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_2)
        conc_4 = Concatenate()([conv_1_2, up_4])
        conv_up_4_1 = Conv2D(32, (3, 3), padding='same')(conc_4)
        conv_up_4_1 = Activation('relu')(conv_up_4_1)
        
        conv_up_4_2 = Conv2D(1, (3, 3), padding='same')(conv_up_4_1)
        result = Activation('sigmoid')(conv_up_4_2)
        
        
        model = Model(inputs=inp, outputs=result)        
        
        '''
        best_w = keras.callbacks.ModelCheckpoint('unet_best.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

        last_w = keras.callbacks.ModelCheckpoint('unet_last.h5',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1)
        
        
        callbacks = [best_w, last_w]
        
        
        
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        
        model.compile(adam, 'binary_crossentropy')
        '''
        
    elif num == 3:
        base_model = ResNet50(weights='imagenet', input_shape=(256,256,3), include_top=False)
         
        base_out = base_model.output
        
        conv1 = base_model.get_layer('activation_134').output
        conv2 = base_model.get_layer('activation_137').output
        conv3 = base_model.get_layer('activation_147').output
        conv4 = base_model.get_layer('activation_160').output
        conv5 = base_model.get_layer('activation_175').output
        
        up1 = UpSampling2D(2, interpolation='bilinear')(conv5)
        conc_1 = Concatenate()([up1, conv4])
        conv_conc_1 = Conv2D(256, (3, 3), padding='same')(conc_1)
        conv_conc_1 = Activation('relu')(conv_conc_1)
        
        up2 = UpSampling2D(2, interpolation='bilinear')(conv_conc_1)
        conc_2 = Concatenate()([up2, conv3])
        conv_conc_2 = Conv2D(128, (3, 3), padding='same')(conc_2)
        conv_conc_2 = Activation('relu')(conv_conc_2)
        
        up3 = UpSampling2D(2, interpolation='bilinear')(conv_conc_2)
        conc_3 = Concatenate()([up3, conv2])
        conv_conc_3 = Conv2D(64, (3, 3), padding='same')(conc_3)
        conv_conc_3 = Activation('relu')(conv_conc_3)
        
        up4 = UpSampling2D(2, interpolation='bilinear')(conv_conc_3)
        conc_4 = Concatenate()([up4, conv1])
        conv_conc_4 = Conv2D(32, (3, 3), padding='same')(conc_4)
        conv_conc_4 = Activation('relu')(conv_conc_4)
        
        up5 = UpSampling2D(2, interpolation='bilinear')(conv_conc_4)
        conv_conc_5 = Conv2D(3, (3, 3), padding='same')(up5)
        conv_conc_5 = Activation('softmax')(conv_conc_5)
        
        '''
        best_w = keras.callbacks.ModelCheckpoint('resnet_best.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

        last_w = keras.callbacks.ModelCheckpoint('resnet_last.h5',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1)
        
        
        callbacks = [best_w, last_w]
        
        
        
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
        
        model.compile(adam, 'binary_crossentropy')
        '''
    
        
    return model
