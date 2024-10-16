import io
import math
import requests
import collections
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers import default_data_collator
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from huggingface_hub import login

# Login using your Hugging Face token
login(token="")

HF_MODEL_PUSH_NAME = "full-finetuned-tweet-4lakh-muril-large"

tokenizer = None
chunk_size = 128
wwm_probability = 0.2
dataset_name = "Anish/tweet-copus"  
model_name = "google/muril-large-cased"  
output_dir = "./muril-base-mlm-output"  
num_train_epochs = 3
batch_size = 64 

# def load_huggingface_data(repo_id):
#     dataset = load_dataset(repo_id)
#     return dataset['train']

def load_huggingface_data():
    url = "https://huggingface.co/datasets/Anish/merged_tweets_corpus_4145473_samples_hindi_plus_nepali.csv/resolve/main/merged_tweets_corpus_4145473_samples_hindi_plus_nepali.csv"
    response = requests.get(url, stream=True)
    response.encoding = 'utf-8'

    # Create an empty list to store the chunks
    all_chunks = []

    # Open the response content with universal newline mode ('U' or 'rU')
    with io.StringIO(response.text, newline='') as csv_file:
        # Read the CSV data using an iterator with a chunk size
        chunksize = 10000  # Adjust chunksize as needed
        for chunk in pd.read_csv(csv_file, chunksize=chunksize, engine='python'):
            # Append each chunk to the list
            all_chunks.append(chunk)

    # Concatenate all chunks into a single DataFrame
    df = pd.concat(all_chunks, ignore_index=True)
    
    return Dataset.from_pandas(df)

def get_pretrained_model(model_checkpoint):
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    return model

def initialized_pretrained_tokenizer(model_checkpoint):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    valid_texts = [text for text in examples["text"] if isinstance(text, str) and text]
    result = tokenizer(valid_texts) 
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def get_tokenized_datasets(dataset):
    tokenized_datasets = dataset.map(
    # tokenize_function, batched=True, remove_columns=["text",  "__index_level_0__"]
    tokenize_function, batched=True, remove_columns=["text"]
    )   

    return tokenized_datasets


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


def fine_tune_mlm(model_name, dataset_name, output_dir, num_train_epochs, batch_size):
    # dataset = load_huggingface_data(dataset_name)
    dataset = load_huggingface_data()
    model = get_pretrained_model(model_name)
    initialized_pretrained_tokenizer(model_name)
    tokenized_datasets = get_tokenized_datasets(dataset)
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    test_size = int(0.04 * len(lm_datasets))
    train_size = len(lm_datasets)-test_size
    
    downsampled_dataset = lm_datasets.train_test_split(train_size=train_size, test_size=test_size, seed=42)

    logging_steps = len(downsampled_dataset["train"]) // batch_size
    model_name = model_name.split("/")[-1]

    training_args = TrainingArguments(
    output_dir=f"{model_name}-{HF_MODEL_PUSH_NAME}",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    logging_steps=logging_steps,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    print(f">>> Before Trainig --> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    for name, param in model.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()


    trainer.train()

    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.push_to_hub()

# Run the fine-tuning process
if __name__ == "__main__":
    # Fine-tune the model
    fine_tune_mlm(model_name, dataset_name, output_dir, num_train_epochs, batch_size)
