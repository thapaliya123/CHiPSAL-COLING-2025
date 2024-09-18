import os
import pandas as pd
from config import (
    base_path, 
    valid_csv_path,
    output_json_file_path
)
from utils import read_json_file
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def convert_target_to_label(target):
    if target == "Individual":
        return 0
    elif target == "Organization":
        return 1
    elif target == "Community":
        return 2


output_json_file_path = os.path.join(base_path, output_json_file_path)
valid_df = pd.read_csv(os.path.join(base_path, valid_csv_path))
valid_df = valid_df[['index', 'cleaned_tweet', 'label']]
valid_df['index'] = valid_df['index'].astype('str')

prediction_content = read_json_file(output_json_file_path)
prediction_df = pd.DataFrame(prediction_content)
prediction_df = prediction_df.transpose().reset_index()

combined_df = pd.merge(valid_df, prediction_df, how = 'inner', on = 'index')
combined_df['target'] = combined_df['target'].apply(convert_target_to_label)

print(confusion_matrix(combined_df['target'], combined_df['label']))
print(classification_report(combined_df['target'], combined_df['label']))
print(accuracy_score(combined_df['target'], combined_df['label']))


