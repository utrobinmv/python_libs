import pickle

def dump_value_to_file(brands_re, filename): 
    '''
    Записывает значение переменной в файл
    '''
    pickle.dump(brands_re, open(filename, 'wb'))
    
    
    