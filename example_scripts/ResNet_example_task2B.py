# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_train.json
# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_dev.json
# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_araieval_24_train_dev.tar.gz


# !tar -xvzf arabic_memes_araieval_24_train_dev.tar.gz

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
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop
from datasets import load_dataset, Dataset, DatasetDict
import torch

import transformers
from transformers import (
    ConvNextFeatureExtractor,
    ResNetConfig,
    ResNetForImageClassification,
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
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    output_dir="./resnet_50/",
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

model_name = 'resnet50'

set_seed(training_args.seed)

import json
#from PIL import Image
import PIL
from datasets import Image
from tqdm import tqdm

# Image.open(obj['img_path']).convert("RGB")

def read_data(fpath, is_test=False):
  if is_test:
    data = {'id': [], 'image': []}
    js_obj = json.load(open(fpath, encoding='utf-8'))
    for obj in tqdm(js_obj):
      data['id'].append(obj['id'])
      data['image'].append(obj['img_path'])
  else:
    data = {'id': [], 'image': [], 'label': []}
    js_obj = json.load(open(fpath, encoding='utf-8'))
    for obj in tqdm(js_obj):
      data['id'].append(obj['id'])
      data['image'].append(obj['img_path'])
      data['label'].append(obj['class_label'])
  return pd.DataFrame.from_dict(data)


l2id = {'not_propaganda': 0, 'propaganda': 1}
train_df = read_data(train_file)
train_df['label'] = train_df['label'].map(l2id)
train_df = Dataset.from_pandas(train_df).cast_column("image", Image())
validation_df = read_data(validation_file)
validation_df['label'] = validation_df['label'].map(l2id)
validation_df = Dataset.from_pandas(validation_df).cast_column("image", Image())
# test_df = read_data(test_file)
# #test_df['label'] = test_df['label'].map(l2id)
# test_df = Dataset.from_pandas(test_df).cast_column("image", Image())



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

config = ResNetConfig(
        num_channels=1,
        layer_type="basic",
        depths=[2, 2],
        hidden_sizes=[32, 64],
        num_labels=num_labels,
)

model = ResNetForImageClassification(config)

feature_extractor = ConvNextFeatureExtractor(
    do_resize=True, do_normalize=False, image_mean=[0.45], image_std=[0.22]
)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_transforms = Compose([Resize(256), CenterCrop(224), ToTensor(), normalize])

def transforms(example_batch):
    """Apply _train_transforms across a batch."""
    # print(example_batch)
    # black and white
    example_batch["pixel_values"] = [_transforms(pil_img.convert("L")) for pil_img in example_batch["image"]]
    return example_batch

if "train" not in raw_datasets:
    raise ValueError("requires a train dataset")
train_dataset = raw_datasets["train"]
if max_train_samples is not None:
    max_train_samples_n = min(len(train_dataset), max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples_n))
train_dataset.set_transform(transforms)

train_dataset

if "validation" not in raw_datasets:
    raise ValueError("requires a validation dataset")
eval_dataset = raw_datasets["validation"]
if max_eval_samples is not None:
    max_eval_samples_n = min(len(eval_dataset), max_eval_samples)
    eval_dataset = eval_dataset.select(range(max_eval_samples_n))
eval_dataset.set_transform(transforms)

if "test" not in raw_datasets and "test_matched" not in raw_datasets:
    raise ValueError("requires a test dataset")
predict_dataset = raw_datasets["test"]
if max_predict_samples is not None:
    max_predict_samples_n = min(len(predict_dataset), max_predict_samples)
    predict_dataset = predict_dataset.select(range(max_predict_samples_n))
predict_dataset = predict_dataset.set_transform(transforms)

for index in random.sample(range(len(train_dataset)), 3):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

metric = evaluate.load("accuracy")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def collate_fn(examples):
    # print(examples)
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
data_collator = collate_fn

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset, # if you have development and test set, uncomment this line
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
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
predict_dataset

id2l = {0:'not_propaganda', 1:'propaganda'}
logger.info("*** Predict ***")
#predict_dataset = predict_dataset.remove_columns("label")
#ids = predict_dataset['id']
#image = predict_dataset['image']
#predict_dataset = predict_dataset.remove_columns("id")
predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
predictions = np.argmax(predictions, axis=1)
output_predict_file = os.path.join(training_args.output_dir, f"task2B_TeamName.tsv")
if trainer.is_world_process_zero():
    with open(output_predict_file, "w") as writer:
        logger.info(f"***** Predict results *****")
        writer.write("id\tlabel\trun_id\n")
        for index, item in enumerate(predictions):
            item = label_list[item]
            item = id2l[item]
            writer.write(f"{predict_dataset[index]['id']}\t{item}\t{model_name}\n")

kwargs = {"finetuned_from": model_name, "tasks": "text-classification"}
trainer.create_model_card(**kwargs)


