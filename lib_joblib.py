from joblib import Parallel, delayed

import pandas as pd
from lib_pandas import find_first_brand

N_JOBS=8

def find_parallel(names_series, brands_re, n_jobs=N_JOBS):
    '''
    Функция разбивает pandas series на n_batches частей, и для каждой части выполняет функцию find_first_brand
    После чего собирает результат в один dataframe
    '''
    
    n_batches = n_jobs
    
    batches = [names_series.iloc[i::n_batches] for i in range(n_batches)]
    
    brand = Parallel(n_jobs=n_jobs)(
        delayed(find_first_brand)(batch, brands_re) 
        for batch in batches #Вызывает в отдельном потоке функцию find_first_brand(batch, brands_re), где batch - это часть фрейма
    )
    brand = pd.concat(brand)
    
    item_brand_mapping = pd.concat([names_series, brand], axis=1, ignore_index=False)
    item_brand_mapping.columns = ['item_name', 'brands']
    
    return item_brand_mapping