# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_train.json
# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_dev.json

# !pip install transformers
# !pip install datasets
# !pip install evaluate
# !pip install --upgrade accelerate

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import datasets
import evaluate
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

train_file = "arabic_memes_propaganda_araieval_24_train.json"
validation_file = "arabic_memes_propaganda_araieval_24_dev.json"
# test_file = 'arabic_memes_propaganda_araieval_24_test.json'

training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir="./distilBERT_m/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    local_rank=1,
    load_best_model_at_end=True,
    save_total_limit=2,
    save_strategy="no",
)

max_train_samples = None
max_eval_samples = None
max_predict_samples = None
max_seq_length = 512
batch_size = 16

transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

model_name = "distilbert-base-multilingual-cased"

set_seed(training_args.seed)

import json


def read_data(fpath, is_test=False):
    if is_test:
        data = {"id": [], "text": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in js_obj:
            data["id"].append(obj["id"])
            data["text"].append(obj["text"])
    else:
        data = {"id": [], "text": [], "label": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in js_obj:
            data["id"].append(obj["id"])
            data["text"].append(obj["text"])
            data["label"].append(obj["class_label"])
    return pd.DataFrame.from_dict(data)


l2id = {"not_propaganda": 0, "propaganda": 1}
train_df = read_data(train_file)
train_df["label"] = train_df["label"].map(l2id)
train_df = Dataset.from_pandas(train_df)
validation_df = read_data(validation_file)
validation_df["label"] = validation_df["label"].map(l2id)
validation_df = Dataset.from_pandas(validation_df)
# test_df = read_data(test_file)
# #test_df['label'] = test_df['label'].map(l2id)
# test_df = Dataset.from_pandas(test_df)


# data_files = {"train": train_df, "validation": validation_df, "test": validation_df}
data_files = {"train": train_df, "validation": validation_df}
for key in data_files.keys():
    logger.info(f"loading a local file for {key}")
raw_datasets = DatasetDict(
    {"train": train_df, "validation": validation_df}  # , "test": test_df
)

# Labels
label_list = raw_datasets["train"].unique("label")
label_list.sort()  # sort the labels for determine
num_labels = len(label_list)
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class LLMWithClassificationHead(nn.Module):
    def __init__(self, model_name, pooling_type, num_classes, hidden_size=768, attention_hidden_size=512, cnn_kernel_size=3):
        super(LLMWithClassificationHead, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling_type = pooling_type
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, attention_hidden_size),
                nn.Tanh(),
                nn.Linear(attention_hidden_size, 1)
            )
        elif pooling_type == "cnn":
            self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2)
        
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling_type == "cls":
            pooled_output = self.cls_pooling(outputs)
        elif self.pooling_type == "max":
            pooled_output = self.max_pooling(outputs)
        elif self.pooling_type == "mean":
            pooled_output = self.mean_pooling(outputs, attention_mask)
        elif self.pooling_type == "attention":
            pooled_output = self.attention_pooling(outputs, attention_mask)
        elif self.pooling_type == "cnn":
            pooled_output = self.cnn_pooling(outputs)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        logits = self.output_layer(pooled_output)        
        return logits
    
    def cls_pooling(self, outputs):
        return outputs.last_hidden_state[:, 0]
    
    def max_pooling(self, outputs):
        return torch.max(outputs.last_hidden_state, dim=1)[0]
    
    def mean_pooling(self, outputs, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def attention_pooling(self, outputs, attention_mask):
        attention_scores = self.attention(outputs.last_hidden_state)
        attention_scores = attention_scores.squeeze(-1)
        attention_scores = attention_scores + (1.0 - attention_mask) * -1e9
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_embeddings = torch.sum(outputs.last_hidden_state * attention_weights.unsqueeze(-1), dim=1)
        return weighted_embeddings
    
    def cnn_pooling(self, outputs):
        last_hidden_state = outputs.last_hidden_state.permute(0, 2, 1)
        cnn_outputs = self.conv1d(last_hidden_state)
        cnn_outputs = F.relu(cnn_outputs)
        pooled_output, _ = torch.max(cnn_outputs, dim=-1)
        return pooled_output

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_labels,
    finetuning_task=None,
    cache_dir=None,
    revision="main",
    use_auth_token=None,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=None,
    use_fast=True,
    revision="main",
    use_auth_token=None,
)

model = LLMWithClassificationHead(model_name=model_name, pooling_type="attention", num_classes=2)

non_label_column_names = [
    name for name in raw_datasets["train"].column_names if name != "label"
]
sentence1_key = non_label_column_names[1]

# Padding strategy
padding = "max_length"

if 128 > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({128}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(128, tokenizer.model_max_length)


def preprocess_function(examples):
    # Tokenize the texts
    args = (examples[sentence1_key],)
    result = tokenizer(
        *args, padding=padding, max_length=max_seq_length, truncation=True
    )

    return result


raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)


if "train" not in raw_datasets:
    raise ValueError("requires a train dataset")
train_dataset = raw_datasets["train"]
if max_train_samples is not None:
    max_train_samples_n = min(len(train_dataset), max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples_n))

train_dataset

if "validation" not in raw_datasets:
    raise ValueError("requires a validation dataset")
eval_dataset = raw_datasets["validation"]
if max_eval_samples is not None:
    max_eval_samples_n = min(len(eval_dataset), max_eval_samples)
    eval_dataset = eval_dataset.select(range(max_eval_samples_n))

# if "test" not in raw_datasets and "test_matched" not in raw_datasets:
#     raise ValueError("requires a test dataset")
# predict_dataset = raw_datasets["test"]
# if max_predict_samples is not None:
#     max_predict_samples_n = min(len(predict_dataset), max_predict_samples)
#     predict_dataset = predict_dataset.select(range(max_predict_samples_n))

predict_dataset = eval_dataset

for index in random.sample(range(len(train_dataset)), 3):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

metric = evaluate.load("accuracy")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


data_collator = default_data_collator

from transformers import TrainerCallback


class CustomCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            logger.info(
                f"Step {state.global_step}: Loss = {state.loss:.4f}, Learning Rate = {state.lr:.2e}, Grad Norm = {state.grad_norm:.2f}"
            )


custom_callback = CustomCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[custom_callback],
)

train_result = trainer.train()
metrics = train_result.metrics
max_train_samples = (
    max_train_samples if max_train_samples is not None else len(train_dataset)
)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))


trainer.save_model()
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

logger.info("*** Evaluate ***")

metrics = trainer.evaluate(eval_dataset=eval_dataset)

max_eval_samples = (
    max_eval_samples if max_eval_samples is not None else len(eval_dataset)
)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# if the test set is available, you don't need to run this cell
predict_dataset = eval_dataset

id2l = {0: "not_propaganda", 1: "propaganda"}
logger.info("*** Predict ***")
# predict_dataset = predict_dataset.remove_columns("label")
ids = predict_dataset["id"]
predict_dataset = predict_dataset.remove_columns("id")
predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
predictions = np.argmax(predictions, axis=1)
output_predict_file = os.path.join(training_args.output_dir, f"task2A_kevinmathew.tsv")
if trainer.is_world_process_zero():
    with open(output_predict_file, "w") as writer:
        logger.info(f"***** Predict results *****")
        writer.write("id\tlabel\trun_id\n")
        for index, item in enumerate(predictions):
            item = label_list[item]
            item = id2l[item]
            writer.write(f"{ids[index]}\t{item}\t{model_name}\n")

ids[0]

kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
trainer.create_model_card(**kwargs)
