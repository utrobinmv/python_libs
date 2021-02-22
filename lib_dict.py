def add_two_dict(dict1, dict2):
    '''
    Функция объединяет ключи двух различных словарей
    получается один общий словарь
    '''
    return dict(list(dict1.items()) + list(dict2.items()))

def in_dictionary(key, dict):
    return key in dict
