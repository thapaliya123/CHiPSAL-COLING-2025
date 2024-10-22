import os
import io
import math
import requests
import collections
import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)
from transformers import default_data_collator, get_scheduler
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from huggingface_hub import get_full_repo_name, Repository


# Login using your Hugging Face token
login(token="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")

HF_MODEL_PUSH_NAME = "muril-base-cased-tweet-grouped-accelerate-epoch-6"
CHECKPOINT_PATH = None
GROUPED_DATA_REPO_ID = "Anish/twitter-devnagari-grouped"
# GPU_NUMBER = 1

tokenizer = None
chunk_size = 128
wwm_probability = 0.2
model_name = "google/muril-base-cased"  
output_dir = "./muril-base-mlm-output"  
num_train_epochs = 6
batch_size = 128

# os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_NUMBER}"

accelerator = Accelerator()

def load_grouped_tokenized_data():
    dataset = load_dataset(GROUPED_DATA_REPO_ID)
    return dataset


def get_pretrained_model(model_checkpoint):
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    return model

def initialized_pretrained_tokenizer(model_checkpoint):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


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


def insert_random_mask(batch):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}



def train_using_acelerate():
    pass


def initialize_repository(output_dir, repo_name):
    # Check if the output directory already exists
    if os.path.exists(output_dir):
        print(f"{output_dir} already exists. Attempting to pull the latest changes.")
        repo = Repository(output_dir)
        try:
            repo.git_pull()  # Pull the latest changes
        except Exception as e:
            print(f"Failed to pull changes: {e}. Re-cloning the repository.")
            # If pull fails, delete the directory and re-clone
            import shutil
            shutil.rmtree(output_dir)
            repo = Repository(output_dir, clone_from=repo_name)
    else:
        # If the directory does not exist, clone the repository
        repo = Repository(output_dir, clone_from=repo_name)
    
    return repo


def fine_tune_mlm(model_name, num_train_epochs, batch_size):
    model = get_pretrained_model(model_name)
    initialized_pretrained_tokenizer(model_name)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    downsampled_dataset = load_grouped_tokenized_data()
    downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    # eval_dataset = downsampled_dataset["test"].map(
    #                                                 insert_random_mask,
    #                                                 batched=True,
    #                                                 remove_columns=downsampled_dataset["test"].column_names,
    #                                             )
    # eval_dataset = eval_dataset.rename_columns(
    #         {
    #             "masked_input_ids": "input_ids",
    #             "masked_attention_mask": "attention_mask",
    #             "masked_labels": "labels",
    #         }
    #     )
    eval_dataset = downsampled_dataset["test"]
    train_dataloader = DataLoader(
        downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    # eval_dataloader = DataLoader(
    #     eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
    # )
    eval_dataloader = DataLoader(
        downsampled_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )


    optimizer = AdamW(model.parameters(),lr=2e-5)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
                                                                    model, optimizer, train_dataloader, eval_dataloader
                                                                    )
    
    
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # repo_name = get_full_repo_name(HF_MODEL_PUSH_NAME)
    # output_dir = HF_MODEL_PUSH_NAME
    # # repo = Repository(output_dir, clone_from=repo_name)
    # if not os.path.exists(output_dir):
    #     rrepo = Repository(output_dir, clone_from=repo_name, use_auth_token=True)
    # else:
    #     repo = Repository(output_dir, use_auth_token=True)
    #     repo.git_pull()

    # repo_name = get_full_repo_name(HF_MODEL_PUSH_NAME)
    # output_dir = HF_MODEL_PUSH_NAME
    # repo = initialize_repository(output_dir, repo_name)

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            unwrapped_model.push_to_hub(
                repo_id=f"Anish/{HF_MODEL_PUSH_NAME}", commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

# Run the fine-tuning process
if __name__ == "__main__":
    # Fine-tune the model
    fine_tune_mlm(model_name, num_train_epochs, batch_size)