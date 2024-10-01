from enums import metric_enums, loss_enums
from transformers import AutoTokenizer

SEED = 42
DEVICE = "cuda"
MAX_LEN = 300
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE=2e-5
# NUM_LABELS = 3
NUM_LABELS = 2
HF_MODEL_PATH = "google/muril-base-cased"
# TRAINING_FILE = "data/taskc/train.csv"
# VALID_FILE = "data/taskc/valid.csv"
TRAINING_FILE = "data/taskb/train.csv"
VALID_FILE = "data/taskb/valid.csv"
TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
MODEL_PATH = f"./models/{HF_MODEL_PATH.split('/')[-1]}"
METRIC_NAME = metric_enums.MetricsEnum.F1_SCORE.value

# For Reproduce
LOSS_FUNCTION = loss_enums.LossFuncEnum.CATEGORICAL_CROSSENTROPY.value
DATA_AUGMENTATION = False 
PREPROCESSING = False


## WANDB LOGGING CONFIG
LOG_TO_WANDB = True
TASK_NAME = TRAINING_FILE.split('/')[1]
TAGS = [HF_MODEL_PATH, TASK_NAME]
WANDB_PROJECT_NAME = "NLP CHIPSAL COLING 2025"


