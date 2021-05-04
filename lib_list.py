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

def add_unique_value_to_list(lst, value):
    '''
    Функция добавляет значение в список если оно уникальное
    '''
    if value not in lst:
        lst.append(value)
        return True
    return False

def sort_str_list_typle(lst):
    '''
    Сортирует список кортежей, по первому элементу
    '''
    
    lst.sort(key = lambda x: (x[0]), reverse = False)
    
def operation_fuction_for_element_list(lst, run_func):
    '''
    Функция вызывает функцию с одним параметром для каждого элемента списка
    '''
    
    return [run_func(i) for i in lst]

def list_unique_values(numbers):
    
    unique_numbers = set(numbers)
    
    return unique_numbers