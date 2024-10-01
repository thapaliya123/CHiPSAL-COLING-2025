from enums import metric_enums, loss_enums
from transformers import AutoTokenizer

# Hyperparameters
SEED = 22
DEVICE = "cuda"
MAX_LEN = 300
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE= 2e-5
NUM_LABELS = 3
METRIC_NAME = metric_enums.MetricsEnum.F1_SCORE.value

# models
# HF_MODEL_PATH = "google/muril-large-cased"
HF_MODEL_PATH = "rahular/varta-bert"
MODEL_PATH = f"./models/{HF_MODEL_PATH.split('/')[-1]}"

# file path
TRAINING_FILE = "data/taskb/train.csv"
VALID_FILE = "data/taskb/valid.csv"

# tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_PATH)

# For Reproducibility and data preprocessing and augmentation
LOSS_FUNCTION = loss_enums.LossFuncEnum.CATEGORICAL_CROSSENTROPY.value
DATA_AUGMENTATION = False 
PREPROCESSING = False

## WANDB LOGGING CONFIG
LOG_TO_WANDB = False
TASK_NAME = TRAINING_FILE.split('/')[1]
TAGS = [HF_MODEL_PATH, TASK_NAME]
WANDB_PROJECT_NAME = "NLP CHIPSAL COLING 2025"