import os
import tempfile

def make_dir(path):
    # define the access rights
    #access_rights = 0o755
    try:
        os.mkdir(path)
        #os.mkdir(path, access_rights)
    except OSError:
        print ("Создать директорию %s не удалось" % path)
    else:
        print ("Успешно создана директория %s " % path)
        
def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Создать директорию %s не удалось" % path)
    else:
        print ("Успешно создана директория %s " % path)
        
def make_tmp_dir():  
    # создаём временную директорию
    with tempfile.TemporaryDirectory() as directory:
        print('Создана временная директория %s' % directory)
    
    # каталог и его содержимое удалены    
    return directory

def dir_exist(path):
    return os.path.exists(path)