import time

def current_time_in_second(): #Возвращает время в секундах
    return round(time.time())

def time_in_second_to_textdate(time_in_second):  #Преобразовывает секунды в текстовую дату
    local_time = time.localtime(time_in_second)
    str_time = time.strftime("%Y_%m_%d_%H-%M-%S", local_time)
    return str_time
