import psycopg2

import pandas as pd

from lib_uuid import return_str_uuid

def pg_db_connect(host, port, user, password, dbname, schema):
    return psycopg2.connect(dbname = dbname, user=user, password=password, port=port, host=host, options="-c search_path=dbo," + schema)

def pg_db_disconnect(conn):
    conn.close()

def pg_db_init(conn):

    cursor = conn.cursor()

    # cursor.execute("""CREATE DATABASE IF NOT EXISTS cards_recognition """)

    # cursor.execute("""DROP TABLE IF EXISTS recognition_edit_answer """)

    #cursor.execute("""CREATE SCHEMA IF NOT EXISTS pkg_recognition
    #    AUTHORIZATION postgres;
    #           """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS recognition_receive
                  (receive_id integer PRIMARY KEY,
                   request_id text,
                   date_time INTEGER,
                   image_name_front text,
                   image_name_back text,
                   message text)
               """)

    cursor.execute("""
                   CREATE SEQUENCE IF NOT EXISTS recognition_receive_seq;
                    ALTER TABLE recognition_receive ALTER receive_id SET DEFAULT NEXTVAL('recognition_receive_seq');
               """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS recognition_send
                  (send_id integer PRIMARY KEY,
                   request_id text,
                   date_time INTEGER,
                   edit_answer INTEGER,
                   success INTEGER,
                   answer text)
               """)

    cursor.execute("""
                   CREATE SEQUENCE IF NOT EXISTS recognition_send_seq;
                    ALTER TABLE recognition_send ALTER send_id SET DEFAULT NEXTVAL('recognition_send_seq');
               """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS recognition_images
                  (image_name text PRIMARY KEY,
                   cloud_id INTEGER)
               """)

    cursor.execute("""
                   CREATE SEQUENCE IF NOT EXISTS recognition_images_seq;
                    ALTER TABLE recognition_images ALTER cloud_id SET DEFAULT NEXTVAL('recognition_images_seq');
               """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS recognition_cloud_image
                  (cloud_id integer PRIMARY KEY,
                   date_time INTEGER,
                   type_image INTEGER,
                   bbox_x1 INTEGER,
                   bbox_y1 INTEGER,
                   bbox_x2 INTEGER,
                   bbox_y2 INTEGER,
                   border_inner INTEGER,
                   border_external INTEGER,
                   image_save_name_original text,
                   check_sum INTEGER,
                   card_x1 INTEGER,
                   card_y1 INTEGER,
                   card_x2 INTEGER,
                   card_y2 INTEGER,
                   image_save_name_card text,
                   answer text)
               """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS recognition_cloud_image_number
                  (image_number_id integer PRIMARY KEY,
                   cloud_id integer,
                   digit_number INTEGER,
                   digit_x1 INTEGER,
                   digit_y1 INTEGER,
                   digit_width INTEGER,
                   digit_height INTEGER,
                   digit text)
               """)

    cursor.execute("""
                   CREATE SEQUENCE IF NOT EXISTS recognition_cloud_image_number_seq;
                    ALTER TABLE recognition_cloud_image_number ALTER cloud_id SET DEFAULT NEXTVAL('recognition_cloud_image_number_seq');
               """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS recognition_cloud_edit_answer
                  (cloud_id integer PRIMARY KEY,
                   date_time INTEGER,
                   correct_recognition INTEGER,
                   correct_recognition_number INTEGER,
                   answer_changed INTEGER,
                   send_answer INTEGER,
                   answer_download INTEGER,
                   template_del INTEGER,
                   answer_prev text,
                   answer_edit text)
               """)


    # correct_recognition: 
    # 1 - все распозналось хорошо
    # 2 - распозналось с ошибкой бренд
    # 3 - ошибка в вырезке объекта (требуется улучшить алгоритм вырезки)
    # 4 - ошибка фотографирования (пользователя)
    # 5 - новый неизвестный бренд

    #correct_recognition_number
    # 0 - распознавание не проводилось
    # 1 - распозналось все хорошо
    # 2 - распозналось с ошибкой номера
    # 4 - ошибка фотографирования (пользователя)

    #Колонка которая добавляет признак удаления шаблона
    cursor.execute(""" ALTER TABLE recognition_cloud_edit_answer ADD COLUMN IF NOT EXISTS template_del INTEGER DEFAULT 0; """)

    #Колонка которая добавляет признак типа изображения
    cursor.execute(""" ALTER TABLE recognition_cloud_image ADD COLUMN IF NOT EXISTS type_image INTEGER DEFAULT 0; """)

    #Колонка признак удаления записи
    cursor.execute(""" ALTER TABLE recognition_cloud_edit_answer ADD COLUMN IF NOT EXISTS remove boolean DEFAULT FALSE; """)
    cursor.execute(""" ALTER TABLE recognition_cloud_image ADD COLUMN IF NOT EXISTS remove boolean DEFAULT FALSE; """)
    cursor.execute(""" ALTER TABLE recognition_images ADD COLUMN IF NOT EXISTS remove boolean DEFAULT FALSE; """)

    
    #type_image = 0 - это фото карты
    #type_image = 1 - это картинка скриншот


    cursor.execute("""
                   ALTER TABLE recognition_cloud_edit_answer ALTER correct_recognition SET DEFAULT 0;
                   ALTER TABLE recognition_cloud_edit_answer ALTER correct_recognition_number SET DEFAULT 0;
                   ALTER TABLE recognition_cloud_edit_answer ALTER answer_changed SET DEFAULT 0;
                   ALTER TABLE recognition_cloud_edit_answer ALTER send_answer SET DEFAULT 0;
                   ALTER TABLE recognition_cloud_edit_answer ALTER answer_download SET DEFAULT 0;
                   ALTER TABLE recognition_send ALTER success SET DEFAULT 0;
               """)

    cursor.execute("""
                   ALTER TABLE recognition_cloud_edit_answer ALTER template_del SET DEFAULT 0;
                   ALTER TABLE recognition_cloud_image ALTER type_image SET DEFAULT 0;
               """)


    conn.commit()

def pg_db_recognition_cloud_edit_answer(con, coud_id, date_time, correct_recognition, correct_recognition_number, send_answer, answer_prev, answer_edit, update_save):

    cursor = con.cursor()

    command = (f"SELECT cloud_id from recognition_cloud_edit_answer where cloud_id = {coud_id}")
    cursor.execute(command)
    results = cursor.fetchall()
    if len(results) == 0:
        command = (f"INSERT INTO recognition_cloud_edit_answer (cloud_id) VALUES ({coud_id});")
        cursor.execute(command)
    elif update_save == False:
        cursor.close()
        #Нельзя перетирать уже имеющийся ответ, при многопользовательской работе
        return
    
    answer_changed = 0
    if answer_prev != answer_edit:
        answer_changed = 1
    
    answer_edit = text_ekran(answer_edit)
    answer_prev = text_ekran(answer_prev)
    
    command = f"UPDATE recognition_cloud_edit_answer SET date_time = {date_time}, correct_recognition = {correct_recognition}, correct_recognition_number = {correct_recognition_number}, send_answer = {send_answer}, answer_changed = {answer_changed}, answer_prev = '{answer_prev}', answer_edit = '{answer_edit}' where cloud_id={coud_id}"
    cursor.execute(command)

    con.commit()

    cursor.close()

def pg_db_recognition_list_edit_answer(con):

    cursor = con.cursor()

    command = (f'''
            select DISTINCT recognition_receive.request_id, recognition_receive.image_name_front, recognition_receive.image_name_back, recognition_receive.message from recognition_receive
                inner join (select DISTINCT image_name from recognition_images where cloud_id in (
                select DISTINCT cloud_id as cl_id
                from recognition_cloud_edit_answer
                where send_answer = 0 and 
                answer_changed = 1 
                and correct_recognition > 0 
                and correct_recognition < 6 )) as t2 
            ON t2.image_name = recognition_receive.image_name_back or 
                t2.image_name = recognition_receive.image_name_front
                ''')
    cursor.execute(command)
    results = cursor.fetchall()
    # if len(results) != 0:
    #     a = 1

    cursor.close()

    return results

def pg_db_recognition_list_card_for_download(con):

    cursor = con.cursor()

    command = (f'''
                  select DISTINCT list_files.cloud_id, list_files.date_time, list_answers.correct_recognition, list_files.image_save_name_original, list_files.image_save_name_card,
                  list_files.card_x1, list_files.card_y1, list_files.card_x2, list_files.card_y2,
                  list_files.bbox_x1, list_files.bbox_y1, list_files.bbox_x2, list_files.bbox_y2, list_files.border_inner, list_files.border_external,
                  list_answers.answer_edit, list_files.type_image 
                  from recognition_cloud_image as list_files
                  inner join (SELECT DISTINCT cloud_id, answer_edit, correct_recognition
	                  FROM recognition_cloud_edit_answer
	                  WHERE correct_recognition != 0 and answer_download = 0) as list_answers
                          ON list_files.cloud_id = list_answers.cloud_id
                ''')
    cursor.execute(command)
    
    df = pd.DataFrame(cursor.fetchall())

    if df.empty == False:
        df.columns = ['cloud_id', 'date_time', 'correct_recognition', 'image_save_name_original', 'image_save_name_card', 'card_x1', 'card_y1', 'card_x2', 'card_y2',
                      'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'border_inner', 'border_external', 'answer', 'type_image']
    
    cursor.close()

    return df


def pg_db_recognition_view_edit_answer(con, image_name):

    answer = None
    cloud_id = None

    cursor = con.cursor()

    command = (f'''
            select answer_edit, cloud_id from recognition_cloud_edit_answer
where cloud_id in (select cloud_id from recognition_images where image_name = '{image_name}' LIMIT 1) LIMIT 1
                ''')
    cursor.execute(command)
    results = cursor.fetchall()
    if len(results) == 0:
         pass
    else:
        answer = results[0][0]
        cloud_id = results[0][1]

    cursor.close()

    return cloud_id, answer

def pg_db_new_receive(con, request_id, date_time, image_name_front, image_name_back, message):

    cursor = con.cursor()

    # command = (f"SELECT request_id from recognition_receive where request_id = '{request_id}'")
    # cursor.execute(command)
    # results = cursor.fetchall()
    # if len(results) == 0:
    #     command = (f"INSERT INTO recognition_receive (request_id) VALUES ('{request_id}');")
    #     cursor.execute(command)
    
    # command = f"UPDATE recognition_receive SET date_time = {date_time}, image_name_front = '{image_name_front}', image_name_back = '{image_name_back}' where request_id='{request_id}'"
    # cursor.execute(command)

    command = (f"INSERT INTO recognition_receive (request_id, date_time, image_name_front, image_name_back, message) VALUES ('{request_id}', {date_time}, '{image_name_front}', '{image_name_back}', '{message}');")
    cursor.execute(command)
    
    con.commit()

    cursor.close()

def pg_db_new_send(con, request_id, date_time, edit_answer, answer, success):

    cursor = con.cursor()

    answer_ekran = text_ekran(answer)

    command = (f"INSERT INTO recognition_send (request_id, date_time, edit_answer, success, answer) VALUES ('{request_id}', {date_time}, {edit_answer}, {success}, '{answer_ekran}');")
    cursor.execute(command)

    con.commit()

    cursor.close()

def pg_db_send_edit_answer(con, cloud_id, answer):

    cursor = con.cursor()

    command = (f"UPDATE recognition_cloud_edit_answer SET send_answer = {answer} where cloud_id = {cloud_id};")
    cursor.execute(command)

    con.commit()

    cursor.close()

def pg_db_send_download_card(con, cloud_id, answer_download):

    cursor = con.cursor()

    command = (f"UPDATE recognition_cloud_edit_answer SET answer_download = {answer_download} where cloud_id = {cloud_id};")
    cursor.execute(command)

    con.commit()

    cursor.close()

def pg_db_get_next_screenshot_recieve_id(con):
    '''
    Запрос служебный поэтому он может быть не очень оптимальным. 
    Вернее он совсем не оптимальный
    '''

    cursor = con.cursor()

    result = -1

    command = (f"SELECT DISTINCT request_id FROM recognition_receive WHERE request_id LIKE '%-%'")
    cursor.execute(command)
    
    results = cursor.fetchall()
    if len(results) == 0:
        pass
    else:
        df = pd.DataFrame(results)
        df.columns = ['request_id']
        
        df = df.sort_values(by='request_id',ascending=False)
        min_res = df['request_id'].values[0]
        min_res = min(0,int(min_res))
        min_res = min_res - 1
        result = min_res

    cursor.close()

    return result


def pg_db_get_cloud_id_recognition_image(con, image_name):

    cursor = con.cursor()

    result = 0

    command = (f"SELECT cloud_id from recognition_images where image_name = '{image_name}'")
    cursor.execute(command)
    results = cursor.fetchall()
    if len(results) == 0:
        result = 0
    else:
        for row in results:
            result = row[0]
            break

    cursor.close()

    return result

def pg_db_set_cloud_id_recognition_image(con, image_name):

    cursor = con.cursor()

    cloud_id = pg_db_get_cloud_id_recognition_image(con, image_name)

    if cloud_id == 0:
            command = (f"INSERT INTO recognition_images (image_name) VALUES ('{image_name}');")
            cursor.execute(command)
            con.commit()

            cloud_id = pg_db_get_cloud_id_recognition_image(con, image_name)

    cursor.close()

    return cloud_id

def pg_db_recognition_cloud_image_get_key(con, id, key):

    cursor = con.cursor()

    result = 0

    command = (f"SELECT {key} from recognition_cloud_image where cloud_id = {id} LIMIT 1")
    cursor.execute(command)
    results = cursor.fetchall()
    if len(results) == 0:
        result = ''
    else:
        for row in results:
            result = row[0]
            break

    cursor.close()

    return result

def text_ekran(answer):
    
    return answer.replace("'","''")
    

def pg_db_recognition_cloud_image_set(con, id, dict):

    cursor = con.cursor()

    command = (f"SELECT cloud_id from recognition_cloud_image where cloud_id = '{id}'")
    cursor.execute(command)
    results = cursor.fetchall()
    if len(results) == 0:
        command = (f"INSERT INTO recognition_cloud_image (cloud_id) VALUES ('{id}');")
        cursor.execute(command)

    type_image = dict['type_image']

    date_time = dict['date_time']
    
    if type_image == 1:
        bbox_x1 = 0
        bbox_y1 = 0
        bbox_x2 = 0
        bbox_y2 = 0
        border_inner = 0
        border_external = 0
        image_save_name_original = dict['image_save_name_original']
        check_sum = dict['check_sum']
        card_x1 = 0
        card_y1 = 0
        card_x2 = 0
        card_y2 = 0
        image_save_name_card = ''
        answer = dict['answer']
    else:
        bbox_x1 = dict['bbox_x1']
        bbox_y1 = dict['bbox_y1']
        bbox_x2 = dict['bbox_x2']
        bbox_y2 = dict['bbox_y2']
        border_inner = dict['border_inner']
        border_external = dict['border_external']
        image_save_name_original = dict['image_save_name_original']
        check_sum = dict['check_sum']
        card_x1 = dict['card_x1']
        card_y1 = dict['card_y1']
        card_x2 = dict['card_x2']
        card_y2 = dict['card_y2']
        image_save_name_card = dict['image_save_name_card']
        answer = dict['answer']
    
    answer = text_ekran(answer)

    command = f"UPDATE recognition_cloud_image SET date_time = {date_time}, type_image = {type_image}, bbox_x1 = {bbox_x1}, bbox_y1 = {bbox_y1}, bbox_x2 = {bbox_x2}, bbox_y2 = {bbox_y2}, border_inner = {border_inner}, border_external = {border_external}, image_save_name_original = '{image_save_name_original}', check_sum = {check_sum}, card_x1 = {card_x1}, card_y1 = {card_y1}, card_x2 = {card_x2}, card_y2 = {card_y2}, image_save_name_card = '{image_save_name_card}', answer = '{answer}' where cloud_id={id}"
    cursor.execute(command)

    con.commit()

    cursor.close()


def pg_db_list_recognition(con):

    cursor = con.cursor()

    command = (f'''
            SELECT recognition_cloud_image.cloud_id,
                   recognition_cloud_image.image_save_name_original,
                   recognition_cloud_image.image_save_name_card,
                   recognition_cloud_image.date_time,
                   recognition_cloud_image.bbox_x1,
                   recognition_cloud_image.bbox_y1,
                   recognition_cloud_image.bbox_x2,
                   recognition_cloud_image.bbox_y2,
                   recognition_cloud_image.border_inner,
                   recognition_cloud_image.border_external,
                   recognition_cloud_image.answer
            FROM recognition_cloud_image
            LEFT JOIN recognition_cloud_edit_answer
                ON recognition_cloud_image.cloud_id = recognition_cloud_edit_answer.cloud_id
            WHERE recognition_cloud_edit_answer.cloud_id IS NULL and recognition_cloud_image.type_image = 0
        ''')
    cursor.execute(command)

    df = pd.DataFrame(cursor.fetchall())

    if df.empty == False:
        df.columns = ['cloud_id', 'image_save_name_original', 'image_save_name_card', 'date_time', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'border_inner', 'border_external', 'answer']

    cursor.close()

    return df

def pg_db_list_recognition_screenshot(con):

    cursor = con.cursor()

    command = (f'''
            SELECT recognition_cloud_image.cloud_id,
                   recognition_cloud_image.image_save_name_original,
                   recognition_cloud_image.image_save_name_card,
                   recognition_cloud_image.date_time,
                   recognition_cloud_image.bbox_x1,
                   recognition_cloud_image.bbox_y1,
                   recognition_cloud_image.bbox_x2,
                   recognition_cloud_image.bbox_y2,
                   recognition_cloud_image.border_inner,
                   recognition_cloud_image.border_external,
                   recognition_cloud_image.answer
            FROM recognition_cloud_image
            LEFT JOIN recognition_cloud_edit_answer
                ON recognition_cloud_image.cloud_id = recognition_cloud_edit_answer.cloud_id
            WHERE recognition_cloud_edit_answer.cloud_id IS NULL and recognition_cloud_image.type_image = 1
        ''')
    cursor.execute(command)

    df = pd.DataFrame(cursor.fetchall())

    if df.empty == False:
        df.columns = ['cloud_id', 'image_save_name_original', 'image_save_name_card', 'date_time', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'border_inner', 'border_external', 'answer']

    cursor.close()

    return df


def pg_db_list_recognition_unsuccess(con, success):
    '''
    success=0 - find success == 0
    success=1 - find success == 1
    '''

    tmp_unsucess_rec = 'tmp_' + return_str_uuid()
    tmp_unsuccess_list_receive = 'tmp_' + return_str_uuid()
    tmp_unsuccess_cloud_id = 'tmp_' + return_str_uuid()
    tmp_quary_cloud_image = 'tmp_' + return_str_uuid()

    cursor = con.cursor()

    command = (f'''
        CREATE TEMP TABLE tmp_unsucess_rec
        AS 
        SELECT request_id
                FROM pkg_recognition.recognition_send
        GROUP BY request_id
        HAVING MAX(success) = {success};
        
        CREATE TEMP TABLE tmp_unsuccess_list_receive
        AS 
        SELECT request_id, image_name_front AS image_name
                FROM pkg_recognition.recognition_receive
                WHERE request_id in (SELECT request_id FROM tmp_unsucess_rec)
        UNION
        SELECT request_id, image_name_back
                FROM pkg_recognition.recognition_receive
                WHERE request_id in (SELECT request_id FROM tmp_unsucess_rec)
                
        ;
        
        CREATE TEMP TABLE tmp_unsuccess_cloud_id
        AS 
        SELECT cloud_id FROM pkg_recognition.recognition_images
        WHERE image_name in (SELECT image_name FROM tmp_unsuccess_list_receive)
        
        ;
        CREATE TEMP TABLE tmp_quary_cloud_image
        AS 
        SELECT recognition_cloud_image.cloud_id,
                   recognition_cloud_image.image_save_name_original,
                   recognition_cloud_image.image_save_name_card,
                   recognition_cloud_image.date_time,
                   recognition_cloud_image.bbox_x1,
                   recognition_cloud_image.bbox_y1,
                   recognition_cloud_image.bbox_x2,
                   recognition_cloud_image.bbox_y2,
                   recognition_cloud_image.border_inner,
                   recognition_cloud_image.border_external,
                   recognition_cloud_image.answer
        FROM pkg_recognition.recognition_cloud_image
        LEFT JOIN pkg_recognition.recognition_cloud_edit_answer
                ON recognition_cloud_image.cloud_id = recognition_cloud_edit_answer.cloud_id
        WHERE recognition_cloud_edit_answer.cloud_id IS NULL and recognition_cloud_image.type_image = 0
        ;
        
        SELECT * FROM tmp_quary_cloud_image
        WHERE cloud_id in (SELECT cloud_id FROM tmp_unsuccess_cloud_id)
        ''').replace('pkg_recognition.','') \
            .replace('tmp_unsucess_rec',tmp_unsucess_rec) \
            .replace('tmp_unsuccess_list_receive',tmp_unsuccess_list_receive) \
            .replace('tmp_unsuccess_cloud_id',tmp_unsuccess_cloud_id) \
            .replace('tmp_quary_cloud_image',tmp_quary_cloud_image) \
            
    cursor.execute(command)

    df = pd.DataFrame(cursor.fetchall())

    command = (f'''
        DROP TABLE IF EXISTS tmp_unsucess_rec;
        DROP TABLE IF EXISTS tmp_unsuccess_list_receive;
        DROP TABLE IF EXISTS tmp_unsuccess_cloud_id;
        DROP TABLE IF EXISTS tmp_quary_cloud_image;
        ''').replace('pkg_recognition.','') \
            .replace('tmp_unsucess_rec',tmp_unsucess_rec) \
            .replace('tmp_unsuccess_list_receive',tmp_unsuccess_list_receive) \
            .replace('tmp_unsuccess_cloud_id',tmp_unsuccess_cloud_id) \
            .replace('tmp_quary_cloud_image',tmp_quary_cloud_image) \
 
    cursor.execute(command)


    if df.empty == False:
        df.columns = ['cloud_id', 'image_save_name_original', 'image_save_name_card', 'date_time', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'border_inner', 'border_external', 'answer']

    cursor.close()

    return df


def pg_db_list_recognition_recover(con):

    cursor = con.cursor()

    command = (f'''
            SELECT recognition_cloud_image.cloud_id,
                   recognition_cloud_image.image_save_name_original,
                   recognition_cloud_image.type_image,
                   recognition_cloud_image.image_save_name_card,
                   cloud_edit_answer.date_time,
                   recognition_cloud_image.bbox_x1,
                   recognition_cloud_image.bbox_y1,
                   recognition_cloud_image.bbox_x2,
                   recognition_cloud_image.bbox_y2,
                   recognition_cloud_image.border_inner,
                   recognition_cloud_image.border_external,
                   cloud_edit_answer.answer_edit AS answer
            FROM pkg_recognition.recognition_cloud_image AS recognition_cloud_image 
            INNER JOIN 	
            (SELECT cloud_id, date_time, answer_edit
                FROM pkg_recognition.recognition_cloud_edit_answer
                ORDER BY date_time DESC
                LIMIT 10) AS cloud_edit_answer
                        ON recognition_cloud_image.cloud_id = cloud_edit_answer.cloud_id    
                ''').replace('pkg_recognition.','')
    cursor.execute(command)

    df = pd.DataFrame(cursor.fetchall())

    if df.empty == False:
        df.columns = ['cloud_id', 'image_save_name_original', 'type_image', 'image_save_name_card', 'date_time', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'border_inner', 'border_external', 'answer']

    cursor.close()

    return df

def pg_db_list_template_for_del(con):
    '''
    Функция возвращает список шаблонов, которые распознаны и прошли через админку и можно удалить их шаблоны из minio
    '''
    cursor = con.cursor()

    command = (f'''
              SELECT image_name, cloud_id
                 FROM recognition_images WHERE cloud_id in (
                 SELECT cloud_id FROM recognition_cloud_edit_answer WHERE template_del = 0);
                ''')
    
    cursor.execute(command)
    
    results = cursor.fetchall()
 
    result = []
    if len(results) != 0:
        for row in results:
            result.append((row[0],row[1]))

    cursor.close()

    return result

def pg_db_make_template_del(con, list_del):
    '''
    Функция возвращает список шаблонов, которые распознаны и прошли через админку и можно удалить их шаблоны из minio
    '''
    cursor = con.cursor()

    macro_comand = ''

    for cloud_id in list_del:
        macro_comand += (f'''UPDATE recognition_cloud_edit_answer SET template_del = 1 WHERE cloud_id = {cloud_id}; ''')
    
    cursor.execute(macro_comand)

    con.commit()
    
    cursor.close()   

def pg_db_list_recive(con):

    cursor = con.cursor()

    #command = (f'''
        #SELECT receive_id, request_id, date_time, message, join_front.cloud_id as cloud_id_front, join_back.cloud_id as cloud_id_back, image_name_front, image_name_back
                #FROM pkg_recognition.recognition_receive
        #LEFT JOIN (SELECT image_name, cloud_id
                #FROM pkg_recognition.recognition_images) AS join_front
                #ON image_name_front = join_front.image_name
        #LEFT JOIN (SELECT image_name, cloud_id
                #FROM pkg_recognition.recognition_images) AS join_back
                #ON image_name_back = join_back.image_name
        #ORDER BY receive_id DESC LIMIT 10
        #''').replace('pkg_recognition.','')
    
    command = (f'''
            SELECT receive_id, pkg_recognition.recognition_receive.request_id, date_time, message, join_front.cloud_id as cloud_id_front, join_back.cloud_id as cloud_id_back, image_name_front, image_name_back, last_answers.answer
                FROM pkg_recognition.recognition_receive
            LEFT JOIN (SELECT image_name, cloud_id
                           FROM pkg_recognition.recognition_images) AS join_front
                ON image_name_front = join_front.image_name
            LEFT JOIN (SELECT image_name, cloud_id
                           FROM pkg_recognition.recognition_images) AS join_back
                ON image_name_back = join_back.image_name
            
                LEFT JOIN (
            
                            SELECT send_last.request_id, send_last.answer FROM (
                            SELECT DISTINCT request_id, max(send_id) as max_send_id
                        FROM pkg_recognition.recognition_send WHERE request_id in (
                             SELECT request_id
                        FROM pkg_recognition.recognition_receive	
                        ORDER BY receive_id DESC LIMIT 20)
                         GROUP BY request_id ) as send_max
                        INNER JOIN (SELECT request_id, send_id, answer
                                            FROM pkg_recognition.recognition_send ORDER BY send_id DESC LIMIT 20) as send_last
            
                                ON send_max.request_id = send_last.request_id AND send_max.max_send_id = send_last.send_id		
                                        ) as last_answers
            
                ON pkg_recognition.recognition_receive.request_id = last_answers.request_id
            
            ORDER BY receive_id DESC LIMIT 10
            
        ''').replace('pkg_recognition.','')
    
    cursor.execute(command)

    df = pd.DataFrame(cursor.fetchall())

    if df.empty == False:
        df.columns = ['receive_id', 'request_id', 'date_time', 'message', 'cloud_id_front', 'cloud_id_back', 'image_name_front', 'image_name_back', 'answer']

    cursor.close()

    return df

def pg_db_list_cards(conn):

    cursor = conn.cursor()

    cursor.execute(f'''SELECT card_name, template_id from card_store_template
                   where is_branded_template is true
                   and is_public_template is true
                   and active is true
                   ORDER BY card_name
                   ''')

    df = pd.DataFrame(cursor.fetchall())
    df.columns = ['card_name', 'template_id']

    cursor.close()

    return df



def pg_db_recognition_state_cloud_edit_answer(con):

    cursor = con.cursor()

    command = (f'''
              SELECT date_time
	         FROM recognition_cloud_edit_answer
              ORDER BY date_time LIMIT 1;
                ''')
    
    cursor.execute(command)
    results = cursor.fetchall()
 
    result = 0
    if len(results) != 0:
        for row in results:
            result = row[0]
            break

    return result

# check_init_db(connect)
