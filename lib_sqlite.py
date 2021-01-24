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


def sqlite_list_card_get_numcode_param(conn, card_name, side, version):

    cursor = conn.cursor()

    cursor.execute(f"SELECT number_code, number_len FROM cards_list WHERE side = {side} AND version = {version} AND card_name = '{card_name}'")

    for row in cursor:
        return (row[0], row[1]) #number_code, number_len
    return (0,0)

def sqlite_list_card_barcodetype_or_none(conn, card_name, side):

    result = None

    cursor = conn.cursor()

    cursor.execute(f"SELECT DISTINCT bar_code_type FROM cards_list WHERE card_name = '{card_name}' and side = {side} and bar_code > 0")



    df = pd.DataFrame(cursor.fetchall())

    cursor.close()

    if len(df) != 0:
        df.columns = ['bar_code_type']

        result = df['bar_code_type']


    return result

