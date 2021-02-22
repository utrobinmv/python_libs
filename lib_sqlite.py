import sqlite3
import pandas as pd

def sqlite_connect(database):
    return sqlite3.connect("cards_dict.db")

def sqlite_list_cards(conn):

    cursor = conn.cursor()

    cursor.execute(f"SELECT card_name, template_id FROM card_store")

    df = pd.DataFrame(cursor.fetchall())
    df.columns = ['card_name', 'template_id']

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

def sqlite_list_card_barcodetype_or_none(conn, card_name, side, all_barcodetype):

    result = None  

    cursor = conn.cursor()
    
    if all_barcodetype == False:
        cursor.execute(f'SELECT DISTINCT bar_code_type FROM cards_list WHERE card_name = "{card_name}" and side = {side} and bar_code > 0')
    else:
        cursor.execute(f"SELECT DISTINCT bar_code_type FROM cards_list WHERE bar_code_type IS NOT NULL")

    df = pd.DataFrame(cursor.fetchall())

    cursor.close()

    if len(df) != 0:
        df.columns = ['bar_code_type']

        result = df['bar_code_type']

    return result

def sqlite_get_card_name(conn, template_id):

    cursor = conn.cursor()

    cursor.execute(f"SELECT card_name FROM card_store WHERE template_id = {template_id}")

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
