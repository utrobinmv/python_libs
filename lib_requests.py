import requests

def get_json_dict_from_url(url):
    r = requests.get(url)
    r_json = r.json()
    return r_json
