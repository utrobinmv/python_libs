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

def dir_is_dir(path):
    return os.path.isdir(path)

def dir_listdir(path):
    
    list_dir = os.listdir(path)
    
    return list_dir  

def save_text_to_file(filename,str_text):
    my_file = open(filename, "w")
    my_file.write(str_text)
    my_file.close()    
    
def filename_without_ext(filename):
    
    full_name = os.path.basename(filename)
    name = os.path.splitext(full_name)[0]    
    
    return name

def read_text_file_with_utf8_start(filename):
    
    #bytes = min(32, os.path.getsize(filename))
    #raw = open(filename, 'rb').read(bytes)
    #if raw.startswith(codecs.BOM_UTF8):
        #encoding = 'utf-8-sig'
    #else:
        #result = chardet.detect(raw)
        #encoding = result['encoding']

    with open(filename, 'r', encoding="utf-8-sig") as f:
        txt_lines = f.read().splitlines()    
        
    return txt_lines
