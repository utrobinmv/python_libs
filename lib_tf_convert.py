import tensorflow as tf
import numpy as np

from sklearn import model_selection

from lib_ml_convert import feutures_categorial_fit, feutures_categorial_transform

def creat_tf_model(number_sub_classes):
    number_classes = 10
    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(units=number_sub_classes*16, activation='relu'),
        tf.keras.layers.Dense(units=number_classes*4, activation='relu'),
        tf.keras.layers.Dense(number_sub_classes, activation='softmax')
        #tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        #tf.keras.layers.Reshape([1, -1]),
    ])
    
    return multi_step_dense

def compile_and_fit(model, train, labels, epochs=20, patience=2):
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                patience=patience,
    #                                                mode='min')
    #model.compile(loss=tf.losses.MeanSquaredError(),
    #            optimizer=tf.optimizers.Adam(),
    #            metrics=[tf.metrics.MeanAbsoluteError()])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    #  history = model.fit(train, epochs=MAX_EPOCHS,
    #                      validation_data=labels,
    #                      callbacks=[early_stopping])

    history = model.fit(train, labels, epochs=epochs)
    return history


def tf_cross_validation(all_data, all_test, number_sub_classes, epochs):

    kf = model_selection.StratifiedKFold(n_splits=5)
    metrics_list = []
    for train_idx, val_idx in kf.split(all_data, all_test):
        train_data = all_data[train_idx]
        train_label = all_test[train_idx]
        test_data = all_data[val_idx]
        test_label = all_test[val_idx]
        print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)
        
        #Создаем модель каждый раз, чтобы она не училась на попытках
        multi_step_dense = creat_tf_model(number_sub_classes)
        
        history = compile_and_fit(multi_step_dense, train_data, train_label, epochs=epochs)
        
        test_loss, test_acc = multi_step_dense.evaluate(test_data, test_label, verbose=2)
        
        del(multi_step_dense)
        
        print('\nТочность на проверочных данных:', test_acc, test_loss)
        metrics_list.append((test_acc, test_loss))
        
    print('Точность кросс-валидации (acc, loss): mean =', np.mean(metrics_list, axis=0), ', min =', np.min(metrics_list, axis=0), ', max =', np.max(metrics_list, axis=0), ', std =',np.std(metrics_list, axis=0))
    
    del(train_data)
    del(train_label)
    del(test_data)
    del(test_label)
    
def tf_run_cross_validation(model_data_train, model_data_label):
    #Крос-валидация tf с учетом того, что данные разбиты на части


    all_parts = 1
    
    dict_label_score = {}
    dict_tf_model = {}
    all_data = []
    all_test = []

    for part in range(all_parts):
        print(' iter ', part+1)
        train_data_part = model_data_train[model_data_train.part == part+1]
        train_label_part = model_data_label[model_data_label.part == part+1]

        train_data_part.drop(['pn_lot_anon'], axis = 1, inplace = True)
        
        train_data_part.drop(['part'], axis = 1, inplace = True)
        train_label_part.drop(['part'], axis = 1, inplace = True)
        
        transform_column = 'inn_label'

        name_dict = 'part' + str(part)

        dict_label_score[name_dict] = feutures_categorial_fit(train_label_part[transform_column])

        train_label_part['inn_sub_label'] = feutures_categorial_transform(dict_label_score[name_dict], train_label_part[transform_column])

        train_label_part.drop([transform_column], axis = 1, inplace = True)

        all_data = train_data_part.values
        all_test = train_label_part.values

        number_sub_classes = len(train_label_part.groupby("inn_sub_label"))

        #dict_tf_model[name_dict] = creat_tf_model(number_sub_classes)

        tf_cross_validation(all_data, all_test, number_sub_classes, 1)

    
    del(train_data_part)
    del(train_label_part)
    del(all_data)
    del(all_test)
    del(dict_label_score)