import os
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

base_path = config.get('Paths', 'base_path')
few_shot_prompt_path = os.path.join(base_path, config.get('Paths', 'few_shot_prompt_path'))
output_json_file_path = os.path.join(base_path, config.get('Paths', 'output_json_file_path'))
train_csv_path = os.path.join(base_path, config.get('Paths', 'train_csv_path'))
valid_csv_path = os.path.join(base_path, config.get('Paths', 'valid_csv_path'))
raw_train_csv_path = os.path.join(base_path, config.get('Paths', 'raw_train_csv_path'))
synthetic_data_path = os.path.join(base_path, config.get('Paths', 'synthetic_data_path'))


api_key = open(os.path.join(base_path, config.get('OpenAI', 'api_key_path')), "r").read().strip('\n')
model_name = config.get('Model', 'model_name')
embedding_model_name = config.get('Model', 'embedding_model_name')
role = config.get('Model', 'role')
max_tokens = int(config.get('Model', 'max_tokens'))
temperature = float(config.get('Model', 'temperature'))
top_p = float(config.get('Model', 'top_p'))
frequency_penalty = float(config.get('Model', 'frequency_penalty'))
presence_penalty = float(config.get('Model', 'presence_penalty'))

