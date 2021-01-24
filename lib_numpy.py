import numpy as np

#Random function
def random_int(max_val):
    '''
    Возвращает целое число от 0 до max_val
    '''
    return int(round(np.random.random() * max_val))

'''
np.random.randint(0, 3, 10)
>>> array([2, 2, 3, 3, 1, 1, 0, 2, 3, 2])
Создать массив из целых чисел от 0 до 3, размер 10
'''


'''
np.random.set_state(state)
np.random.random() одно число от 0 до 1
np.random.random(10) #Создает массив из 10 числе от 0 до 1
'''

'''
np.all(a==2) - Проверяет, все ли элементы массива вдоль данной оси имеют значение True.

np.all(train_data.registered + train_data.casual == train_labels)
Проверяет что сумма двух колонок pandas df равна значению из третьей колонки, либо массиву numpy
'''
