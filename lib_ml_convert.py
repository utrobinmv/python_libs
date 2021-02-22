import numpy as np
import pandas as pd

from sklearn import preprocessing

from sklearn import decomposition

from sklearn import ensemble

from sklearn import metrics

from sklearn import model_selection

def reshape_for_scaler(data_do):
    data = np.array(data_do, dtype='float32')
    data = data.reshape(-1,1)
    return data

def pandas_series_object_to_numpy(data_series):
    '''
    Конвертирует pandas data series в numpy
    '''
    new_dtype = data_series[0].dtype
    result_list = []
    
    for el in data_series:
        result_list.append(el)
        
    result = np.array(result_list, dtype = new_dtype)
        
    return result

def features_fit(dict_models, dataset, columns_categorical, columns_text_vector, columns_numeric_data):
    '''
    Обучает для конвертации данные
    Пример вызова
    dict_models = {}
    columns_categorical = ['fz']
    columns_text_vector = text_columns_vector #Сюда передаются не сами колонки а уже преобразованные вектора от них большой размероностью
    columns_numeric_data = ['lot_price', 'region_code', 'okpd2_int']
    features_fit(dict_models, mlka_train_data, columns_categorical, columns_text_vector, columns_numeric_data)
    features_transform(dict_models, mlka_train_data, columns_categorical, columns_text_vector, columns_numeric_data)
    '''
    for column in columns_categorical:
        dict_models[column] = preprocessing.LabelEncoder()
        data = dataset[column]
        dict_models[column].fit(data)
    for column in columns_categorical:
        dict_models[column + '_label'] = preprocessing.StandardScaler()
        data = reshape_for_scaler(dict_models[column].transform(dataset[column]))
        dict_models[column + '_label'].fit(data)
    for column in columns_text_vector:
        dict_models[column] = decomposition.PCA(n_components=2)
        data = pandas_series_object_to_numpy(dataset[column].values)
        dict_models[column].fit(data)
    for column in columns_numeric_data:
        dict_models[column] = preprocessing.StandardScaler()
        data = reshape_for_scaler(dataset[column])
        dict_models[column].fit(data)
        
def features_transform(dict_models, dataset, columns_categorical, columns_text_vector, columns_numeric_data):
    '''
    Конвертирует колонки dataframe
    '''
    for column in columns_categorical:
        data = dataset[column]
        dataset[column + '_label'] = dict_models[column].transform(data)
    for column in columns_categorical:
        data = reshape_for_scaler(dataset[column + '_label'])
        dataset[column + '_label_x'] = dict_models[column + '_label'].transform(data)
    for column in columns_text_vector:
        data = pandas_series_object_to_numpy(dataset[column].values)
        p_comps = dict_models[column].transform(data)
        dataset[column + '_x'] = p_comps[:,0]
        dataset[column + '_y'] = p_comps[:,1]
    for column in columns_numeric_data:
        data = reshape_for_scaler(dataset[column])
        dataset[column + '_x'] = dict_models[column].transform(data)
        
        
def feutures_categorial_fit(dataset):
    enc = preprocessing.LabelEncoder()
    enc.fit(dataset)
    print('Количество различных классов:',enc.classes_.shape) #Количество различных контрагентов
    return enc

def feutures_categorial_transform(enc, dataset):
    transform_dataset = enc.transform(dataset)
    return transform_dataset


def scalar_inn_on_parts(model_data_train, model_data_label):
    '''
    Процедура конвертирует метки классов в скалярные числа
    При условии что весь датасет разделен на части
    dict_label_encoders = scalar_inn_on_parts()
    '''
    
    all_parts = int(max(model_data_train['part']))
    dict_label_score = {}
    
    df_all_label = 0

    for part in range(all_parts):
        print(' iter ', part+1)
        train_data_part = model_data_train[model_data_train.part == part+1]
        train_label_part = model_data_label[model_data_label.part == part+1]

        transform_column = 'participant_inn_kpp_anon'

        name_dict = 'part' + str(part)

        dict_label_score[name_dict] = feutures_categorial_fit(train_label_part[transform_column])

        train_label_part['inn_label'] = feutures_categorial_transform(dict_label_score[name_dict], train_label_part[transform_column])

        if part == 0:
            df_all_label = train_label_part
        else:
            df_all_label = df_all_label.append(train_label_part)
            

    model_data_label['inn_label'] = df_all_label['inn_label']
    model_data_label.drop([transform_column], axis = 1, inplace = True)
    
    return dict_label_score


def model_cross_val_score_shuffle_metrics(classifier, train_data, train_label, all_parts):

    scorer = metrics.make_scorer(metrics.accuracy_score)
    #scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=True) #Метрика greater_is_better говорит, что чем больше значение метрики тем лучше

    cv_strategy = model_selection.StratifiedKFold(n_splits=5)
    cv_strategy.get_n_splits(train_label)
    scoring = model_selection.cross_val_score(classifier, train_data, train_label, scoring = scorer, cv = cv_strategy)

    print('Ridge mean:{}, max:{}, min:{}, std:{}'.format(scoring.mean(), scoring.max(), 
                                                         scoring.min(), scoring.std()))

    del(cv_strategy)
    
    
def select_model(model_data_train, model_data_label, num):

    all_parts = int(max(model_data_train['part']))

    for part in range(all_parts):
        print(' iter ', part+1)
        train_data_part = model_data_train[model_data_train.part == part+1]
        train_label_part = model_data_label[model_data_label.part == part+1]
        
        train_data_part.drop(['part'], axis = 1, inplace = True)
        train_label_part.drop(['part'], axis = 1, inplace = True)

        train_data_part.drop(['pn_lot_anon'], axis = 1, inplace = True)
        
        #if num == 1:
            #estimator = linear_model.SGDClassifier(loss = 'log', random_state = 1, max_iter=1000)
        #elif num == 2:
            #estimator = linear_model.RidgeClassifier(random_state = 1)
        #elif num == 3:
            #estimator = tree.DecisionTreeClassifier(random_state = 1, max_depth = 50)
        #elif num == 4:
            #estimator = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 50, random_state = 1)
        #elif num == 5:
            #estimator = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 1)
        #elif num == 6:
            #estimator = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators = 50, min_child_weight=3)
        #elif num == 7:
            #estimator = neighbors.KNeighborsClassifier(n_neighbors = 35)
            
        estimator = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 50, random_state = 1)

        model_cross_val_score_shuffle_metrics(estimator, train_data_part, train_label_part, all_parts)
        
        del(estimator)
        
def optimal_parameter_model(model_data_train, model_data_label, num):


    all_parts = int(max(model_data_train['part']))
    for part in range(all_parts):
        print(' iter ', part+1)
        train_data_part = model_data_train[model_data_train.part == part+1]
        train_label_part = model_data_label[model_data_label.part == part+1]
        
        train_data_part.drop(['part'], axis = 1, inplace = True)
        train_label_part.drop(['part'], axis = 1, inplace = True)

        train_data_part.drop(['pn_lot_anon'], axis = 1, inplace = True)
        
        #if num == 4:
        estimator = ensemble.RandomForestClassifier(n_estimators = 50, random_state = 1)
            
        parameters_grid = {
          'max_depth' : [20, 40, 50, 60],
          'bootstrap' : [True, False],
          'max_features' : ['sqrt', 'log2', None],
          }
            
            
            
        scorer = metrics.make_scorer(metrics.accuracy_score)
        #scorer = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=True) #Метрика greater_is_better говорит, что чем больше значение метрики тем лучше

        cv_strategy = model_selection.StratifiedKFold(n_splits=5)
        cv_strategy.get_n_splits(train_label_part)
        
        grid_cv = model_selection.GridSearchCV(estimator, parameters_grid, scoring = scorer, cv = cv_strategy)
        
        grid_cv.fit(train_data_part, train_label_part)
        
        print(grid_cv.best_score_) #Лучший результат
        
        print(grid_cv.best_params_) #Список лучших значений параметров


def train_model(model_data_train, model_data_label, model_data_test, num):
    dict_estimator = {}
    predict_answers = {}
    
    all_parts = int(max(model_data_train['part']))
    
    for part in range(all_parts):
        print(' iter ', part+1)
        train_data_part = model_data_train[model_data_train.part == part+1]
        train_label_part = model_data_label[model_data_label.part == part+1]
        
        train_data_part.drop(['part'], axis = 1, inplace = True)
        train_label_part.drop(['part'], axis = 1, inplace = True)
        
        number_sub_classes = len(train_data_part.groupby("pn_lot_anon"))

        train_data_part.drop(['pn_lot_anon'], axis = 1, inplace = True)
        
        name_dict = 'part' + str(part)

        #if num == 1:
            #dict_estimator[name_dict] = linear_model.SGDClassifier(loss = 'log', random_state = 1, max_iter=1000)
        #elif num == 3:
            #dict_estimator[name_dict] = tree.DecisionTreeClassifier(random_state = 1, max_depth = 50)
        #elif num == 4:
            ##dict_estimator[name_dict] = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 20, random_state = 1)
            #pass
        #elif num == 7:
            #dict_estimator[name_dict] = neighbors.KNeighborsClassifier(n_neighbors = 200)
            
        #dict_estimator[name_dict].fit(train_data_part, train_label_part)
        
        est = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 20, bootstrap = False, max_features = 'sqrt', random_state = 1)
        est.fit(train_data_part, train_label_part)
        
        #Сразу запустим predict, для экономии оперативной памяти!
        test_data_part = model_data_test[model_data_test.part == part+1]
        test_data_part.drop(['part'], axis = 1, inplace = True)
        test_data_part.drop(['pn_lot_anon'], axis = 1, inplace = True)

        predict_answer = est.predict_proba(test_data_part)
        predict_answers[name_dict] = predict_answer
        
    #return dict_estimator
    return predict_answers

def decoder_predict_answers(predict_answers, dict_label_encoders, model_data_test):
    '''
    Функция расшифровывает ответы из predict_answers
    и добавляет колонку label - строковую
    
    dict_label_encoders = scalar_inn_on_parts()
    '''
    
    
    all_parts = 1
    ### Код расшифровывает ответы из predict_answers, и превращает их в dataframe all_part_result_df
    
    all_part_result_df = 0
    
    for part in range(all_parts):
        print(' iter ', part+1)
    
        name_dict = 'part' + str(part)
    
        res_df = pd.DataFrame(predict_answers[name_dict])
    
        res_df['pn_lot_anon'] = pd.DataFrame({'pn_lot_anon':model_data_test[model_data_test.part == part+1]['pn_lot_anon'].values})
        res_df['region_code_x'] = pd.DataFrame({'region_code_x':model_data_test[model_data_test.part == part+1]['region_code_x'].values})
        res_df['okpd2_int_x'] = pd.DataFrame({'okpd2_int_x':model_data_test[model_data_test.part == part+1]['okpd2_int_x'].values})
    
        columns = list(res_df)
        columns.pop()
        columns.pop()
        columns.pop()
    
        df_all_column = 0
    
        for idx, name_col in enumerate(columns):
            df = res_df[[name_col, 'pn_lot_anon', 'region_code_x', 'okpd2_int_x']]
            df['inn_label'] = name_col
            df.rename(columns = {name_col:'score'}, inplace = True)
    
            if idx == 0:
                df_all_column = df
            else:
                df_all_column = df_all_column.append(df)
    
        df_all_column['participant_inn_kpp_anon'] = dict_label_encoders[name_dict].inverse_transform(df_all_column.inn_label)
    
        df_filter = df_all_column[df_all_column.score != 0]
    
        df_filter = df_filter[['pn_lot_anon', 'participant_inn_kpp_anon', 'score', 'region_code_x', 'okpd2_int_x']]
    
        if part == 0:
            all_part_result_df = df_filter
        else:
            all_part_result_df = all_part_result_df.append(df_filter)    