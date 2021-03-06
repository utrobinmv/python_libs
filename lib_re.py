import re

def multi_str_re(strings, debug=True, borders=True):
    '''
    Преобразует массив строк strings, в одно большое регулярное выражение, на поиск этих строк
    поиск происходит на границе слов
    
    Пример вызова:
    
    brands_re = multi_str_re(
    brands_unique.sort_values(by='count', ascending=False).index, 
    borders=True, 
    debug=True)
    
    '''
    
    re_str = '|'.join(
        [re.escape(s) for s in strings]
    )
    if borders:
        re_str = r'\b(?:' + re_str + r')(?!\S)'
        
    if debug:
        print(re_str)
    return re.compile(re_str, re.UNICODE)


def pandas_series_find_all_re(names_series, brands_re):
    '''
    Функция возвращает series c индексами и значениями из ReExpression brands_re
    '''
    return names_series.str.findall(brands_re)

def pandas_series_replace_all_re(names_series, brands_re):
    '''
    Функция заменяет все из регулярного выражения
    '''
    return names_series.str.replace(brands_re,'')

def findall_with_str(re_expr, str_text):
    
    return re.findall(re_expr, str_text)