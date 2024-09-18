import os
import json
import pandas as pd
from tqdm import tqdm

from prompt_handler import PromptHandler
from utils import read_json_file, write_json_file
from hate_speech_detector import HateSpeechDetector

from config import (
    role,
    top_p,
    api_key,
    max_tokens,
    model_name,
    temperature,
    presence_penalty,
    frequency_penalty,
)
from config import (
    base_path,
    train_csv_path,
    valid_csv_path,
    few_shot_prompt_path,
    output_json_file_path,
)


def clean_json_string(json_str):
    # Example to remove or replace invalid characters
    cleaned_str = re.sub(r"u00xD7", "*", json_str)  # Replace invalid character
    return cleaned_str


if __name__ == "__main__":

    api_response = {}
    few_shot_prompt_path = os.path.join(base_path, few_shot_prompt_path)
    output_json_file_path = os.path.join(base_path, output_json_file_path)
    valid_df = pd.read_csv(os.path.join(base_path, valid_csv_path))
    prompt_handler = PromptHandler()
    hate_speech_deector = HateSpeechDetector()


    for index, tweet in tqdm(valid_df[['index', 'cleaned_tweet']].values):
        sys_prompt = prompt_handler.gen_prompt_template(sentence=tweet)
        example_prompt = prompt_handler.generate_prompt_examples(few_shot_prompt_path)
        complete_prompt = sys_prompt + "\n" + example_prompt
        try:
            output_response = hate_speech_deector.identify_hatespeech_target(complete_prompt)
            api_response[index] = output_response
        except Exception as e:
            print(f"Exception Occurred in {pdf_file_path} and error is {e}")
    write_json_file(data=api_response, file_path=output_json_file_path)
