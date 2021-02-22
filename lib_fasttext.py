import numpy as np
import pandas as pd
import fasttext

def load_model():
    '''
    Функция загружает обученную модель
    '''
    #Модели можно скачать, в открытом доступе с сайта
    #https://fasttext.cc/docs/en/crawl-vectors.html
    ft = fasttext.load_model('models/cc.ru.300.bin')
    print(ft.get_dimension())
    
    return ft

def get_embeddings(sent, model):
    '''
    Функция конвертирует предложение в вектор
    '''
    vect = model.get_sentence_vector(sent) if not pd.isna(sent) else np.zeros(model.get_dimension())
    return vect

def futures_text_transform(ft, dataset, text_columns):
    '''
    Конвертирует список колонок text_columns
    '''
    vector_text_columns = []
    for text_colum in text_columns:
        name_column = text_colum + '_vector'
        dataset[name_column] = dataset[text_colum].apply(lambda x: get_embeddings(x, model=ft))
        vector_text_columns.append(name_column)
    return vector_text_columns