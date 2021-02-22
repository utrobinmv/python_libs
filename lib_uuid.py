import uuid

def return_uuid4():
    return uuid.uuid4()

def return_str_uuid():
    return str(return_uuid4()).replace('-','')
