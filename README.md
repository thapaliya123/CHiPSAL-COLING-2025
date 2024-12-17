# CHiPSAL@COLING-2025 (MDS Bots)

Welcome to the CHIPSAL-COLING-2025 project! This repository contains the code and resources for Shared task on Natural Language Understanding of Devanagari Script Languages at [CHiPSAL@COLING 2025](https://sites.google.com/view/chipsal/home).
> The "Shared task on Natural Language Understanding of Devanagari Script Languages" at CHIPSAL@COLING 2025 focuses on addressing key challenges in processing Devanagari-scripted languages. In multilingual contexts, accurate language identification is critical, making the first subtask, Devanagari Script Language Identification, essential for identifying whether a given text is in Devanagari script. Hate speech detection is another significant aspect of understanding social dynamics, especially within online spaces. Subtask B, Hate Speech Detection, aims to determine whether a given text contains hate speech, with annotated datasets marking the presence or absence of such content. Building on this, Subtask C, Targets of Hate Speech Identification, focuses on identifying specific targets of hate speech, such as individuals, organizations, or communities. This shared task facilitates comprehensive Devanagari Script Language understanding, targeting key challenges in script identification, hate speech detection, and the identification of hate speech targets.

# Getting Started
> To get started with the project, clone the repository and follow the steps:

## 1. **Configuration**  
Setup the desired configuration via updating `config.py`

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
```python
python3 train.py | python3 train.py --gpu-number 0
```

## 3. **Inference Command**
**3.1 Without Ensemble Learning**
```python
# Command
python3 predict.py --model-path <model-path-here> --test-data-path <test-path-here> --submission-file-path <submission-csv-file-path>

# Example
python3 predict.py --model-path ./models/muril-base-cased-f1_score-0.6789460853867482.bin --test-data-path ./data/taskc/test.csv --submission-file-path ./submissions/submission.json
```

**3.2 With Ensemble Learning**

```python
# Command
python3 predict.py --model-path <should-be-model-directory>  --test-data-path ./data/taskc/test.csv  --submission-file-path ./submissions/test.json --ensemble soft_vote

# Example
python3 predict.py --model-path ./models/ensemble_models --test-data-path ./data/taskc/test.csv  --submission-file-path ./submissions/test.json --ensemble soft_vote
```

# LeaderBoard Results
> Our team named **MDS Bots** achieved 1st, 3rd, and 6th positions on shared Tasks C, B, and A respectively.
## [Shared Task C](https://codalab.lisn.upsaclay.fr/competitions/20000#results)
<img src="assets/1.png" width=500 height=500>

## [Shared Task B](https://codalab.lisn.upsaclay.fr/competitions/20000#results)

<img src="assets/2.png" width=500 height=500>

## [Shared Task A](https://codalab.lisn.upsaclay.fr/competitions/20000#results)
<img src="assets/3.png" width=500 height=500>

# For Citations
`Will release later` 

# Status
## Updating...

# Important Links  
1. **Shared Task Details**
    - https://codalab.lisn.upsaclay.fr/competitions/20000#learn_the_details

2. **Datasets (Shared Task A, B, C)**
    - https://github.com/therealthapa/chipsal24?tab=readme-ov-file

**_MDSBots@2024_**
