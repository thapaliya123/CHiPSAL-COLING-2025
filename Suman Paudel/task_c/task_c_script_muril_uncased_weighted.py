import os
import logging
import gdown
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          Trainer, 
                          TrainingArguments, 
                          EarlyStoppingCallback
                          )
from torch.nn import CrossEntropyLoss  # Add this import

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

def download_file(file_id: str, target_file_name: str):
    """Download a file from Google Drive using its file ID."""
    dataset_url = f"https://drive.google.com/u/1/uc?id={file_id}&export=download"
    gdown.download(dataset_url, target_file_name)
    logging.info(f"File '{target_file_name}' downloaded successfully.")

def load_data(file_names: str):
    """Load datasets from CSV files."""
    col_names = ['index', 'text', 'label']
    train_df = pd.read_csv(file_names[0], header=0, names=col_names)
    valid_df_tweet = pd.read_csv(file_names[1])
    valid_df_label = pd.read_csv(file_names[2])
    
    valid_df = pd.merge(valid_df_tweet, valid_df_label, on='index')
    valid_df.columns = col_names
    train_df.drop('index', axis=1, inplace=True)
    valid_df.drop('index', axis=1, inplace=True)
    
    return train_df, valid_df

def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize the dataset with dynamic max length."""
    return dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=max_length), batched=True)

def find_max_token_length(df, tokenizer):
    """Finds the max token length based on the dataset."""
    lengths = [len(tokenizer.encode(text)) for text in df['text'].values]
    return int(np.percentile(lengths, 98))

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get the labels from inputs
        labels = inputs.pop("labels")
        device = labels.device
        # Forward pass through the model
        outputs = model(**inputs)

        # Get the logits (predictions) from the model output
        logits = outputs.logits

        # Compute the weighted loss
        class_weights = torch.tensor([2.0615, 2.5864, 7.7958], dtype=torch.float, device=device)
        weighted_loss_fn = CrossEntropyLoss(weight=class_weights)
        loss = weighted_loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

def main():
    # File IDs and names
    file_ids = [
        '166k7N9KV6jEDvvAr9iLTwrWyfdEcAs5p', 
        '1-2TjS6xPfjWj9YaJGSf-JXXXfNz-2pNT', 
        '1-1k1yHOGP7Wij1mUG2iKaSTN8i1WUgPz'
    ]
    file_names = ["train.csv", "val_tweet.csv", "val_label.csv"]

    # Download files
    for file_id, file_name in zip(file_ids, file_names):
        download_file(file_id, file_name)

    # Load datasets
    train_df, valid_df = load_data(file_names)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/muril-large-cased")


    # Find dynamic max token length
    max_length = find_max_token_length(train_df, tokenizer)
    logging.info(f"Dynamic max length for tokenization: {max_length}")

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # Tokenization
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length)
    valid_dataset = tokenize_dataset(valid_dataset, tokenizer, max_length)

    # DataLoader (Increased batch size for H100 GPU)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32)

    # Model training
    model = AutoModelForSequenceClassification.from_pretrained("google/muril-large-cased", num_labels=3)

    os.environ["WANDB_DISABLED"] = "true"

    # Training arguments with evaluation strategy and best model saving
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,  # Adjusted number of epochs
        per_device_train_batch_size=32,  # Adjusted batch size
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=1,
        save_strategy="steps",
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=100,
        fp16=True,
        gradient_accumulation_steps=1,  # Adjusted based on increased batch size
        warmup_steps=500,
        weight_decay=0.01,
        report_to="none",  # Disabling unnecessary reports like WandB
        logging_first_step=True,
    )

    # Initialize the Custom Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early stopping
        
    )

    # Ensure model parameters are contiguous
    for name, param in model.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    trainer.train()

    # Predictions and evaluation
    y_true = valid_dataset['label']
    predictions = trainer.predict(valid_dataset)
    y_hat = np.argmax(predictions.predictions, axis=1)

    # Save metrics to CSV
    metrics_df = pd.DataFrame({'y_true': y_true, 'y_hat': y_hat})
    metrics_df.to_csv('metrics.csv', index=False)
    logging.info("Metrics saved to metrics.csv")

    # Save classification report
    report = classification_report(y_true, y_hat, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv')
    logging.info("Classification report saved to classification_report.csv")

    # Log GPU memory usage
    logging.info(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    main()