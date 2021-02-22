import pandas as pd

'''
#Проверяем есть ли отличия значений в колонках
mlka_dataset.loc[mlka_dataset['fz_label'] != mlka_dataset['fz']] 
'''


def create_dataframe(data, columns):
    df = pd.DataFrame(data)
    if len(df) != 0:
        #df.columns = ['category_name', 'name_box', 'image_filename', 'bbox']
        df.columns = columns
    return df

def return_line_by_index(data, index):
    return data.iloc[index]

#def create_dataframe(data):
    #return pd.DataFrame(data)

def change_label_on_classes(data, label_classes_list):
    '''
    Замена числовых классов, на строковые
    Функция обходит построчно данные data (type Pandas Series) и заменяет аналогичные из списка сопоставления из label_classes_list
    '''
    return data.apply(lambda x: label_classes_list[x])

def return_filter_lines_on_value(data_frame, data_frame_column_series, value_filter):
    '''
    Филтрует фрейм по конкретному значению в колонке
    data_frame - фрейм данных
    data_frame_column_series - колонка данных (type Pandas Series)
    value_filter - конкретное значение в выбранной колонке
    '''
    return data_frame[data_frame_column_series == value_filter]

#Рисование графиков в pandas

def visible_histogram(df, **args):
    '''
    Выводит гистограмму на экран по значениям колонки column_name
    где y - количество раз которое встречается данное значение
        x - это само значение
    Параметы (column=None, by=None, grid=True, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False, figsize=None, layout=None, bins=10, backend=None, legend=False, **kwargs)
    '''
    return df.hist(**args)

def pandas_series_to_dict(series):
    '''
    Преобразует pandas.series в dict
    '''
    return series.to_dict()

def pandas_dataframe_to_list(df):
    '''
    Перебирает в цикле все строки dataframe, и превращает их в массив словарей
    '''
    
    list_dict = []
    
    for idx, rows in df.iterrows():
        #print(rows['cloud_id'])
        list_dict.append(pandas_series_to_dict(rows))

    return list_dict

def dataframe_read_csv(filename):
    return pd.read_csv(filename, sep=';')

def dataframe_save_csv(filename, df):
    recommendation = pd.DataFrame(df, columns=['inn_kpp', 'actual_recommended_pn_lot', 'similarity_score'])
    recommendation.to_csv(filename, index=False, sep=';')    

def dataframe_add_column(train_data):
    '''
    Функция по условию построчно проверяет если None подставляет значение из другой колонки
    '''
    train_data['okpd2_or_additional_code'] = train_data[['okpd2_code', 'additional_code']].apply(lambda x: x[0] if x[1] == 'None' else x[1], axis=1)
    
def dataframe_merge(train_data, train_labels):
    '''
    Объединяет два датафрейма по полю pn_lot_anon
    '''
    inn_kpp_history = pd.merge(train_labels, train_data[['pn_lot_anon','region_code', 'okpd2_or_additional_code']], on=['pn_lot_anon'])

def dataframe_random_35_procedure(train_data, train_labels):
    '''
    Функция выбирает 35 случайных процедур из 
    '''
    inn_kpp_history = pd.merge(train_labels, train_data[['pn_lot_anon','region_code', 'okpd2_or_additional_code']], on=['pn_lot_anon'])
    #группировка по поставщику 
    inn_kpp_history = inn_kpp_history.groupby('participant_inn_kpp_anon').apply(lambda x: [
        list(x['pn_lot_anon']),
        list(x['is_winner']), 
        list(x['fz']), 
        list(x['region_code']), 
        list(x['okpd2_or_additional_code'])]).apply(pd.Series)
    inn_kpp_history = inn_kpp_history.reset_index()
    inn_kpp_history.columns = ['participant_pn_lot_anon', 'list_participant_inn_kpp_anon',
                               'list_is_winner', 'list_fz', 'list_region_code',
                               'list_okpd2_or_additional_code']
    inn_kpp_recommendation = []
    similarity_score = 1
    for inn_kpp in tqdm_notebook(inn_kpp_history.values):
        participant_inn_kpp_anon, list_participant_inn_kpp_anon, list_is_winner, list_fz, list_region_code, list_okpd2_or_additional_code = inn_kpp
        #подвыборка с совпадением региона и ОКПД2 кода актуальной с историей поставщика
        recommendation = test_data[test_data['region_code'].isin(list_region_code) & test_data['okpd2_or_additional_code'].isin(list_okpd2_or_additional_code)]
        if recommendation.shape[0] >= 35:
            #выбор 35 случайных актуальных процедур из подвыборки
            recommendation = recommendation.sample(35)['pn_lot_anon'].values
            for actual_pn_lot in recommendation:
                inn_kpp_recommendation.append([participant_inn_kpp_anon, actual_pn_lot, similarity_score])    
    
def dataframe_div_part(mlka_train_data):
    '''
    Функция разбивает весь датасет на части, и создает колонку part с номером части датасета
    Разбивка в примере выполняется по полям ['region_code_x', 'okpd2_int']
    '''
    
    
    data_for_parts = mlka_train_data[['region_code_x', 'okpd2_int']]
    data_for_parts = data_for_parts.sort_values(by=['region_code_x', 'okpd2_int'])
    print(data_for_parts.value_counts(sort=True)) #Показывает размер каждой части
    mlka_train_data['part'] = int(0)
    mlka_train_data['part'] = mlka_train_data['part'].astype(int)
    
    list_value_counts = data_for_parts.value_counts(sort=False)
    
    droblenie = 5000
    
    #Разьиваем весь dataframe на части, примерно равной величине (в зависимости от данных)
    num_sum = 0
    num_part = int(0)
    
    old_region_code_x = 0
    old_okpd2_int = 0
    
    list_val = []
    list_part = []
    
    for val, num in list_value_counts.iteritems():
        
        region_code_x = val[0]
        okpd2_new = str(val[1])
        
        #print(okpd2_new)    
        
        okpd2_int = int(float(okpd2_new))
        
        #print(okpd2_int)
        
        next_part = False
        
        if old_region_code_x != region_code_x:
            next_part = True
        
        elif old_okpd2_int != okpd2_int:
            next_part = True
        
        elif num_sum + num > droblenie:
            next_part = True
        
        if next_part == False:
            num_sum += num
        else:
            num_part +=1
        
        list_part.append(num_part)
        list_val.append(val)
        
        old_region_code_x = region_code_x
        old_okpd2_int = okpd2_int    
    
        def set_value_part(df, list_part, list_val):
            
            def set_part(region_code_x, okpd2_int):
                '''
                Функция возвращает часть, в которой относится запись
                '''
        
                #okpd2_prb = int(float(okpd2_int))    
            
                val = (region_code_x, okpd2_int)
                
                try:
                    idx = list_val.index(val)
                    return list_part[idx]
                except:
                    return 0
                
                return 0
    
    
            df['part'] = df[['region_code_x', 'okpd2_int']].apply(lambda x: set_part(x[0], x[1]), axis=1)    

        set_value_part(mlka_train_data, list_part, list_val)


def dataframe_sort(mlka_train_data):
    '''
    Сортирует датафрейм
    '''
    mlka_train_data.sort_values(by=['min_publish_date'], inplace = True)

def dataframe_fillna(test_data):
    '''
    Заполняет все NAN в None
    '''
    return test_data.fillna('None')

def dataframe_find_null(df):
    '''
    Функция проверяет, есть в данных df значение null хотя бы в одной из ячеек
    Ответ: 
     True - null встречается
     False - null не встречается
    '''
    return df.isnull().values.any()

def dataframe_find_na(df):
    '''
    Функция проверяет, есть в данных df значение NaN хотя бы в одной из ячеек
    Ответ: 
     True - null встречается
     False - null не встречается
    '''
    return df.isna().values.any()

def dataframe_info(df):
    '''
    Выводит информацию по df - колонки таблицы их имена, и длина, количество записей и прочее
    '''
    return df.info()


def series_value_counts(train_data):
    '''
    Посмотреть частоту значений
    '''
    #Просмотр частоты значений 
    print(train_data['okpd2_or_additional_code'].value_counts())        


def dataframe_delete_rows(df, rows_list):
    '''
    Функция удаляет заданные колонки из df
    Пример использования оригинала: train_data.drop(['datetime', 'count'], axis = 1)
    '''
    return df.drop([rows_list], axis = 1)

def dataframe_from_list_rows(df, rows_list):
    '''
    Возвращает dataframe состоящий только из колонок rows_list
    Пример использования оригинала: new_data = train_data['temp', 'atemp', 'humidity']
    '''
    return df[rows_list]

def row_data_str_to_datetime(row_data):
    '''
    Преобразует значения в конкретной колонке, из строки в формат типа дата и время
    Пример значения в колонке: 2011-01-01 11:00:00
    Пример вызова:
      df.datetime = row_data_str_to_datetime(df.datetime_str)
    Пример оригинала:
      df.datetime = df.datetime_str.apply(pd.to_datetime)
      
    Примеры дальнейшего использования дат:
       row_data['hour'] = row_data.datetime.apply(lambda x: x.hour) #Добавление колонки с часом из данных даты и времени
       row_data['month'] = row_data.datetime.apply(lambda x: x.month) #Добавление колонки с месяцев из данных даты и времени
    
      
    '''
    return row_data.apply(pd.to_datetime)

def row_data_result_function(function, row_data):
    '''
    Функция возвращает результат функции применимым к колонке
    Пример использования оригинала: train_data.datetime.max()
    '''
    if function == 'max':
        return row_data.max()
    elif function == 'min':
        return row_data.min()

def row_data_to_ndarray(row_data_series):
    '''
    Функция возвращает текстовые значения колонки в виде ndarray
    Пример использования оригинала: train_data['count'].values
    '''
    return row_data_series.values


def split_dataframe_two_part(df, num):
    '''
    Данная функция разбивает общий дата фрейм на два куска, путем отделения последних num строк в отдельный dataframe
    Возврат: train_data, learn_data
    '''
    train_data = df.iloc[:-num,:]
    test_data = df.iloc[-num:,:]
    
    return train_data, test_data



def pandas_series_find_all_re(names_series, brands_re):
    '''
    Функция возвращает series c индексами и значениями из RegExpression brands_re
    '''
    return names_series.str.findall(brands_re)



def find_first_brand(names_series, brands_re):
    '''
    Функция вызывает процедуру pandas_series_find_all_re, и отбирает только самые первые из найденных RegExpression
    ''' 
    
    brands = pandas_series_find_all_re(names_series, brands_re)
    first_brand = brands.map(
        lambda b: b[0] if len(b) > 0 else 'отсутствие бренда'
    )
    return first_brand

def dataframe_split_to_n_part(df, n_batches):
    '''
    Функция разбивает dataframe на all_part равных частей и отдает в виде списка частей
    Пример использования:
        dataframe_split_to_n_part(df_sample_test, 8)
    '''
    
    batches = [df.iloc[i::n_batches] for i in range(n_batches)]
    
    return batches



def dataframe_add_id_column(df):
    '''
    Процедура добавляет в dataframe колонку с поэлеменотной нумерацией id
    '''
    
    df['id'] = range(df.shape[0])

    
def dataframe_to_parquet(df, filename_parquet):
    '''
    Процедура записывает в файл значение dataframe (паркует файл)
    '''
    
    df.to_parquet(filename_parquet)


#Мое творчество
def add_two_unique_row(df, class_label_column):
    '''
    Функция добавляет еще по одному уникальному значению класса в датасет, для того чтобы можно было воспользоваться функцией StratifiedShuffleSplit
    в некотором смысле несколько искажает датасет нереальными данным, зато позволяет применить к данным функцию StratifiedShuffleSplit
    
    Пример использования: 
    unikum_inn = mlka_dataset.groupby("participant_inn_kpp_anon").filter(lambda x: len(x) == 1)
    mlka_dataset = mlka_dataset.append(unikum_inn)
    
    '''
    unique_rows = df.groupby(class_label_column).filter(lambda x: len(x) == 1)
    return df.append(unique_rows)
    