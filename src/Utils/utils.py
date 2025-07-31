import sys
sys.path.append("")

from functools import wraps
import time
import os
import yaml
import platform
import subprocess
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from typing import Optional

from dotenv import load_dotenv
load_dotenv()


        
def convert_ms_to_hms(ms):
    seconds = ms / 1000
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    seconds = round(seconds, 2)
    
    return f"{int(hours)}:{int(minutes):02d}:{seconds:05.2f}"


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        time_delta = convert_ms_to_hms(total_time*1000)

        print(f'{func.__name__.title()} Took {time_delta}')
        return result
    return timeit_wrapper

def is_file(path:str):
    return '.' in path

def check_path(path):
    # Extract the last element from the path
    last_element = os.path.basename(path)
    if is_file(last_element):
        # If it's a file, get the directory part of the path
        folder_path = os.path.dirname(path)

        # Check if the directory exists, create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create new folder path: {folder_path}")
        return path
    else:
        # If it's not a file, it's a directory path
        # Check if the directory exists, create it if not
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create new path: {path}")
        return path

def read_config(path = 'config/config.yaml'):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def download_file(url: str, filename: str) -> bool:
    """
    Downloads a file from the given URL and saves it to the specified filename.
    
    Args:
        url (str): The URL of the file to download
        filename (str): The local path to save the downloaded file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {url} to {filename}, size: {os.path.getsize(filename)} bytes")
        if os.path.exists(filename):
            print(f"File {filename} exists")
        return True
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except OSError as e:
        print(f"Error writing file {filename}: {e}")
        return False

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        
    Returns:
        int: Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]



class GoogleTranslator:
    def __init__(self):
        pass

    def translate(self, text, to_lang, max_input_length=4900):
        url = 'https://translate.googleapis.com/translate_a/single'
        
        def get_translation_chunk(chunk):
            params = {
                'client': 'gtx',
                'sl': 'auto',
                'tl': to_lang,
                'dt': ['t', 'bd'],
                'dj': '1',
                'source': 'popup5',
                'q': chunk
            }
            response = requests.get(url, params=params, verify=False).json()
            sentences = response['sentences']
            translated_chunk = ""
            for sentence in sentences:
                translated_chunk += sentence['trans']
            return translated_chunk
        
        if len(text) <= max_input_length:
            return get_translation_chunk(text)
        
        translated_text = ""
        for i in range(0, len(text), max_input_length):
            chunk = text[i:i + max_input_length]
            translated_text += get_translation_chunk(chunk) + "\n"
        
        return translated_text.strip()