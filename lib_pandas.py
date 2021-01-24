import pandas as pd

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

def dataframe_find_null(df):
    '''
    Функция проверяет, есть в данных df значение null хотя бы в одной из ячеек
    Ответ: 
     True - null встречается
     False - null не встречается
    '''
    return df.isnull().values.any()

def dataframe_info(df):
    '''
    Выводит информацию по df - колонки таблицы их имена, и длина, количество записей и прочее
    '''
    return df.info()

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



    