def sorting_str_list(lst):
    '''
    Функция сортирует список строк
    '''
    return sorted(lst)

def sorting_str_list_lower(lst):
    '''
    Функция сортирует список строк
    При этом регистр букв не учитывается.
    '''
    return sorted(lst, key=str.lower, reverse=False)
    
def operation_fuction_for_element_list(lst, run_func):
    '''
    Функция вызывает функцию с одним параметром для каждого элемента списка
    '''
    
    return [run_func(i) for i in lst]