import time
import datetime

def current_time_in_second(): 
    #Возвращает время в секундах
    return round(time.time())

def time_in_second_to_textdate(time_in_second):  
    #Преобразовывает секунды в текстовую дату
    local_time = time.localtime(time_in_second)
    str_time = time.strftime("%Y_%m_%d_%H-%M-%S", local_time)
    return str_time

def current_day_startday_in_second(): 
    #Возвращает время начала дня в секундах по местному времени
    currentdate = datetime.datetime.today()
    currentdate = currentdate.replace(hour=0, minute=0, second=0, microsecond=0)
    time_s = time.mktime(currentdate.timetuple()) + currentdate.microsecond / 1E6
    return round(time_s)



