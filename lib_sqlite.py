import sqlite3
import pandas as pd

def sqlite_connect(database):
    return sqlite3.connect(database)

def sqlite_list_cards(conn):

    cursor = conn.cursor()

    cursor.execute(f"SELECT card_name, hash_id, idx FROM card_store")

    df = pd.DataFrame(cursor.fetchall())
    df.columns = ['card_name', 'hash_id', 'idx']

    cursor.close()

    return df

def sqlite_list_cards_re_text(conn):

    cursor = conn.cursor()

    cursor.execute(f"SELECT card_name, text_find FROM card_texts")

    df = pd.DataFrame(cursor.fetchall())
    df.columns = ['card_name_back', 'text_find']

    cursor.close()

    return df

def sqlite_list_card_get_card_param(conn, card_name, side, version):

    result = None

    cursor = conn.cursor()
    
    if version == 0:
        cursor.execute(f"SELECT max(number_code), max(number_len), max(number_find), max(bar_code) FROM cards_list WHERE side = {side} AND card_name = '{card_name}'")
    else:
        cursor.execute(f"SELECT number_code, number_len, number_find, bar_code FROM cards_list WHERE side = {side} AND version = {version} AND card_name = '{card_name}'")

    df = pd.DataFrame(cursor.fetchall())

    if len(df) != 0:
        df.columns = ['number_code','number_len','number_find','bar_code']

        result = df.iloc[0].to_dict()
        if result['bar_code'] is None:
            result['bar_code'] = 0
        if result['number_find'] is None:
            result['number_find'] = 0
        if result['number_len'] is None:
            result['number_len'] = 0
        if result['number_code'] is None:
            result['number_code'] = 0
              
    return result

def sqlite_list_card_barcodetype(conn, card_name, side, all_barcodetype):

    result = []

    cursor = conn.cursor()
    
    if all_barcodetype == False:
        cursor.execute(f'SELECT DISTINCT bar_code_type FROM cards_list WHERE card_name = "{card_name}" and side = {side} and bar_code > 0')
    else:
        cursor.execute(f"SELECT DISTINCT bar_code_type FROM cards_list WHERE bar_code_type IS NOT NULL")

    df = pd.DataFrame(cursor.fetchall())

    cursor.close()

    if len(df) != 0:
        df.columns = ['bar_code_type']

        result = list(df['bar_code_type'].values)

    return result

def sqlite_get_card_name(conn, hash_id):

    cursor = conn.cursor()

    cursor.execute(f"SELECT card_name FROM card_store WHERE hash_id = {hash_id}")

    for row in cursor:
        return row[0]
    return ""



#def sqlite_get_numcode_param(conn, card_name, side):

    #result = None

    #cursor = conn.cursor()

    #cursor.execute(f"SELECT max(number_len), max(number_find) FROM cards_list WHERE side = {side} AND card_name = '{card_name}'")

    #df = pd.DataFrame(cursor.fetchall())

    #if len(df) != 0:
        #df.columns = ['number_len','number_find']

        #result = df.iloc[0].to_dict()
    
    #return result

###sql_lite_learn
    
def db_find_card_or_none(conn, name_card, side, version):
    cursor = conn.cursor()

    cursor.execute(f"SELECT card_idx FROM cards_list WHERE card_name = '{name_card}' AND side = {side} AND version = {version} LIMIT 1")

    for row in cursor:
        return row[0]

    return None    
    
    
def db_find_or_add_card(conn, name_card, side, version):

    find_result = db_find_card_or_none(conn, name_card, side, version)
    if find_result:
        return find_result

    db_add_card(conn, name_card, side, version)

    return db_find_card_or_none(conn, name_card, side, version)

def db_add_card(conn, name_card, side, version):
    cursor = conn.cursor()

    cursor.execute(f"INSERT INTO cards_list (card_name, side, version, learn, card_check) VALUES ('{name_card}', {side}, {version}, 1, 0)")

    conn.commit()
    
def db_find_card_store_or_none(conn, name_card):
    cursor = conn.cursor()

    cursor.execute(f"SELECT card_name FROM card_store WHERE card_name = '{name_card}' LIMIT 1")

    for row in cursor:
        return row[0]

    return None
    
def db_find_card_id(conn, name_card):
    cursor = conn.cursor()

    cursor.execute(f"SELECT idx FROM card_store WHERE card_name = '{name_card}' LIMIT 1")

    for row in cursor:
        return row[0]

    raise Exception("name_card " + name_card + " не найдено!")
    
def db_find_or_add_card_store(conn, name_card, name_card_transliterate):

    find_result = db_find_card_store_or_none(conn, name_card)
    if find_result:
        return find_result

    db_add_card_store(conn, name_card, name_card_transliterate)

    return db_find_card_store_or_none(conn, name_card)

def db_change_type_numcode_by_idx(conn, idx_card, number_code, number_len, text_card):

    cursor = conn.cursor()

    cursor.execute(f"UPDATE cards_list SET number_code = {number_code}, number_len = {number_len}, text_card = '{text_card}' WHERE card_idx = {idx_card}")

    conn.commit()
    
def db_add_card_store(conn, name_card, name_card_transliterate):
    cursor = conn.cursor()

    name_card_lad = name_card_transliterate

    cursor.execute(f"INSERT INTO card_store (card_name, card_ru, card_lat) VALUES ('{name_card}', '{name_card}', '{name_card_lad}')")

    conn.commit()