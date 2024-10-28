## [Codalab Competition](https://codalab.lisn.upsaclay.fr/competitions/20000#results)

### Notes
- CUDA out of Memory
    - `get_process_id:` nvidia-smi
    - `kill_process_id:` kill -9 <PID>
- Training Command:
    - python train.py

- Training command with specific GPU
    - python train.py --gpu-number <available-gpu-number>
    - python train.py --gpu-number 0
    - python train.py --gpu-number 1



## EVALUATION (Notice from Organizers)
- The submission will be evaluated with a f1-score
- The script takes one prediction file as the input. Your submission file must be a JSON file which is then zipped. We will only take the first file in the zip folder, so do not zip multiple files together.
- for the subtask C, the final prediction submissions should be like the following. Make sure that your individual, organization, and community labels are given as "0", "1", and "2" respectively.
- **Important:** The index in JSON should be in ascending order
```
{"index": 50001, "prediction": 0}
{"index": 50010, "prediction": 1}
{"index": 50074, "prediction": 2}
```
- **Submission File Generation**
```python
import json

# Example prediction data (index and predictions)
predictions = [
    {"index": 50001, "prediction": 0},
    {"index": 50010, "prediction": 1},
    {"index": 50074, "prediction": 2},
]

# Prepare the file in JSON lines format
with open('submission.json', 'w') as f:
    for pred in predictions:
        f.write(json.dumps(pred) + '\n')  # Ensure each line is a separate JSON object


# FINALLY ZIP the file.
zip submission.zip submission.json
```

## Generate Prediction
```
##1. Command (Without Ensemble Learning)
python3 predict.py --model-path <model-path-here> --test-data-path <test-path-here> --submission-file-path <submission-csv-file-path>

### Example
python3 predict.py --model-path ./models/muril-base-cased-f1_score-0.6789460853867482.bin --test-data-path ./data/taskc/test.csv --submission-file-path ./submissions/submission.json

##2. Command (With Ensemble Learning)
python3 predict.py --model-path <should-be-model-directory>  --test-data-path ./data/taskc/test.csv  --submission-file-path ./submissions/test.json --ensemble soft_vote

### Example
python3 predict.py --model-path ./models/ensemble_models --test-data-path ./data/taskc/test.csv  --submission-file-path ./submissions/test.json --ensemble soft_vote
```

## Ensemble Learning
- **Soft Voting:**
```
python3 ensemble.py --model-dir <models-dir>   --soft-vote

Example:
python3 ensemble.py --model-dir ./models/ensemble_models   --soft-vote
```

- **Hard Voting:**
```
python3 ensemble.py --model-dir <models-dir>

Example: 
python3 ensemble.py --model-dir ./models/ensemble_models 
```

## Stacking Prediction
```
python3 stacking_classifier.py --seed 42 --gpu-number 0 --best-model-dir ./models/ensemble_models --train-data-csv ./data/taskc/train_decode_emoji.csv --valid-data-csv ./data/taskc/valid_decode_emoji.csv --test-data-csv ./data/taskc/test.csv --submission-file-path ./submissions/test.json

python3 stacking_classifier.py --seed 42 --gpu-number 0 --best-model-dir ./models/ensemble_models --train-data-csv ./data/taskc/train_decode_emoji.csv --valid-data-csv ./data/taskc/valid_decode_emoji.csv --test-data-csv ./data/taskc/test.csv --submission-file-path ./submissions/test.json
```

