# CHIPSAL-COLING-2025

Welcome to the CHIPSAL-COLING-2025 project! This repository contains the code and resources for competitions. 
## [Codalab Competition](https://codalab.lisn.upsaclay.fr/competitions/20000#results)

## Branches

- **[taska](https://github.com/thapaliya123/CHIPSAL-COLING-2025/tree/taska)**: This branch contains the implementation for Task A.
- **[taskb](https://github.com/thapaliya123/CHIPSAL-COLING-2025/tree/taskb)**: This branch includes the implementation for Task B.
- **[taskc](https://github.com/thapaliya123/CHIPSAL-COLING-2025/tree/taskc)**: This branch features the implementation for Task C, building upon the foundation laid in Task B.

## Getting Started

To get started with the project, clone the repository and check out the desired branch:


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


## Inference Command
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


## Configuration
### For Task A
```python3
TRAINING_FILE = <path to competition train csv file for task A>
VALID_FILE = <path to competition valid csv file for task A>
```
### For Task B
```python3
TRAINING_FILE = <path to competition train csv file for task B>
VALID_FILE = <path to competition valid csv file for task B>
```
### For Task C
```python3
TRAINING_FILE = <path to competition train csv file for task C>
VALID_FILE = <path to competition valid csv file for task C>
```

