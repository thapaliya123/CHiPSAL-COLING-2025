# CHIPSAL-COLING-2025

Welcome to the CHIPSAL-COLING-2025 project! This repository contains the code and resources for Shared task on Natural Language Understanding of Devanagari Script Languages at [CHIPSAL@COLING 2025](https://sites.google.com/view/chipsal/home).
> The "Shared task on Natural Language Understanding of Devanagari Script Languages" at CHIPSAL@COLING 2025 focuses on addressing key challenges in processing Devanagari-scripted languages. In multilingual contexts, accurate language identification is critical, making the first subtask, Devanagari Script Language Identification, essential for identifying whether a given text is in Devanagari script. Hate speech detection is another significant aspect of understanding social dynamics, especially within online spaces. Subtask B, Hate Speech Detection, aims to determine whether a given text contains hate speech, with annotated datasets marking the presence or absence of such content. Building on this, Subtask C, Targets of Hate Speech Identification, focuses on identifying specific targets of hate speech, such as individuals, organizations, or communities. This shared task facilitates comprehensive Devanagari Script Language understanding, targeting key challenges in script identification, hate speech detection, and the identification of hate speech targets.

## [Codalab Competition](https://codalab.lisn.upsaclay.fr/competitions/20000#results)


# Getting Started

To get started with the project, clone the repository and follow the steps:

## 1. **Configuration**  
> Setup the desired configuration via `config.py`

#### For Task A
```python3
TRAINING_FILE = <path to competition train csv file for task A>
VALID_FILE = <path to competition valid csv file for task A>
```
#### For Task B
```python3
TRAINING_FILE = <path to competition train csv file for task B>
VALID_FILE = <path to competition valid csv file for task B>
```
#### For Task C
```python3
TRAINING_FILE = <path to competition train csv file for task C>
VALID_FILE = <path to competition valid csv file for task C>
```

**To Get competition dataset: [click here](https://github.com/therealthapa/chipsal24?tab=readme-ov-file)**

## 2. **Training Command**  
    2.1 `Task A`:

    2.2 `Task B`:
    
    2.3 `Task C`:

## 3. **Inference Command**
```python3
##1. Command (Without Ensemble Learning)
python3 predict.py --model-path <model-path-here> --test-data-path <test-path-here> --submission-file-path <submission-csv-file-path>

### Example
python3 predict.py --model-path ./models/muril-base-cased-f1_score-0.6789460853867482.bin --test-data-path ./data/taskc/test.csv --submission-file-path ./submissions/submission.json

##2. Command (With Ensemble Learning)
python3 predict.py --model-path <should-be-model-directory>  --test-data-path ./data/taskc/test.csv  --submission-file-path ./submissions/test.json --ensemble soft_vote

### Example
python3 predict.py --model-path ./models/ensemble_models --test-data-path ./data/taskc/test.csv  --submission-file-path ./submissions/test.json --ensemble soft_vote
```


**_MDSBots@2024_**