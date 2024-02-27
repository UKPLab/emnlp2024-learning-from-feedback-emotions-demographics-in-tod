'''
    utils.py -- Script contains utility methods.
'''

from typing import Dict, List
from datetime import datetime
import json

def extract_response(response: Dict[str, str]) -> List[str]:
    if 'choices' in response:
        if len(response['choices']) > 0 and 'message' in response['choices'][0]:
            if 'content' in response['choices'][0]['message']:
                return response['choices'][0]['message']['content']\
                    .split('\n\n')
    raise Exception('######## No valid response!\n{}'.format(response))

def get_datetime_now():
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%y%m%d-%H%M%S")
    # E.g., 230417-130459 2023/04/17 13:04:59
    return dt_string

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def load_text(filepath):
    with open(filepath, 'r') as f:
        data = [l.strip() for l in f if len(l.strip()) > 0]
    return data

def dump_json(filepath, data: Dict):
    with open(filepath, 'w') as f:
        json.dump(data, f)