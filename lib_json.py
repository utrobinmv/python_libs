import json

def json_to_dict(json_str):
    return json.loads(json_str)

def dict_to_json(dict_str):
    return json.dumps(dict_str, ensure_ascii=False)

def dict_to_file(filename, json_data):
    with open(filename, "w") as write_file:
        json.dump(json_data, write_file)    