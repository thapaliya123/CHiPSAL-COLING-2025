import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import config
import pandas as pd

# Assuming you have the labels in a numpy array
  # Your training labels

df_train = pd.read_csv(config.TRAINING_FILE)
df_valid = pd.read_csv(config.VALID_FILE)

y_train = df_train['label'].values 
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
