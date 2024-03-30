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

train_file = 'arabic_memes_propaganda_araieval_24_train.json'
validation_file = 'arabic_memes_propaganda_araieval_24_dev.json'
# test_file = 'arabic_memes_propaganda_araieval_24_test.json'

training_args = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir="./distilBERT_m/",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    local_rank= 1,
    load_best_model_at_end=True,
    save_total_limit=2,
    save_strategy="no"
)

max_train_samples = None
max_eval_samples=None
max_predict_samples=None
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

model_name = 'distilbert-base-multilingual-cased'

set_seed(training_args.seed)

import json
def read_data(fpath, is_test=False):
  if is_test:
    data = {'id': [], 'text': []}
    js_obj = json.load(open(fpath, encoding='utf-8'))
    for obj in js_obj:
      data['id'].append(obj['id'])
      data['text'].append(obj['text'])
  else:
    data = {'id': [], 'text': [], 'label': []}
    js_obj = json.load(open(fpath, encoding='utf-8'))
    for obj in js_obj:
      data['id'].append(obj['id'])
      data['text'].append(obj['text'])
      data['label'].append(obj['class_label'])
  return pd.DataFrame.from_dict(data)


l2id = {'not_propaganda': 0, 'propaganda': 1}
train_df = read_data(train_file)
train_df['label'] = train_df['label'].map(l2id)
train_df = Dataset.from_pandas(train_df)
validation_df = read_data(validation_file)
validation_df['label'] = validation_df['label'].map(l2id)
validation_df = Dataset.from_pandas(validation_df)
# test_df = read_data(test_file)
# #test_df['label'] = test_df['label'].map(l2id)
# test_df = Dataset.from_pandas(test_df)



#data_files = {"train": train_df, "validation": validation_df, "test": validation_df}
data_files = {"train": train_df, "validation": validation_df}
for key in data_files.keys():
    logger.info(f"loading a local file for {key}")
raw_datasets = DatasetDict(
    {"train": train_df, "validation": validation_df} # , "test": test_df
)

# Labels
label_list = raw_datasets["train"].unique("label")
label_list.sort()  # sort the labels for determine
num_labels = len(label_list)

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

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    from_tf=bool(".ckpt" in model_name),
    config=config,
    cache_dir=None,
    revision="main",
    use_auth_token=None,
    ignore_mismatched_sizes=False,
)

non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
sentence1_key= non_label_column_names[1]

# Padding strategy
padding = "max_length"

# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
            "\nIgnoring the model labels as a result.",)

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

if 128 > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({128}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.")
max_seq_length = min(128, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],))
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset, # if you have development and test set, uncomment this line
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
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

id2l = {0:'not_propaganda', 1:'propaganda'}
logger.info("*** Predict ***")
# predict_dataset = predict_dataset.remove_columns("label")
ids = predict_dataset['id']
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


