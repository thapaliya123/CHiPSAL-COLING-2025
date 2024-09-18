import re
import json

def remove_multiple_spaces(text):
    """
    Replaces multiple spaces with a single space in the given text.

    Parameters:
    text (str): The input string with potential multiple spaces.

    Returns:
    str: The modified string with single spaces.
    """
    return re.sub(r"\s+", " ", text)


def read_json_file(json_file_path: str)->json: 
    with open(json_file_path, 'r') as json_file:
        json_content = json.load(json_file)
    return json_content


def write_json_file(data, file_path):
    """
    Write a dictionary to a JSON file.

    Parameters:
    - data (dict): The dictionary to be written to the JSON file.
    - file_path (str): The path of the file where the JSON data will be saved.

    Returns:
    - None
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"JSON file '{file_path}' created successfully.")
    except IOError as e:
        print(f"Error writing file: {e}")
