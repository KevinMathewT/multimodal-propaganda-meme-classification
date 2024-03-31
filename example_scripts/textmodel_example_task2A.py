# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_train.json
# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_dev.json
# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_araieval_24_train_dev.tar.gz

# !tar -xvzf arabic_memes_araieval_24_train_dev.tar.gz

# !pip install transformers
# !pip install datasets
# !pip install evaluate
# !pip install --upgrade accelerate
from torch.cuda.amp import autocast, GradScaler

USE_FP16 = True  # Set to False for normal training
if USE_FP16:
    scaler = GradScaler()
else:
    scaler = None

learning_rate = 1e-4
num_train_epochs = 5
train_max_seq_len = 512
max_train_samples = None
max_eval_samples = None
max_predict_samples = None
batch_size = 8
best_macro_f1 = 0.0

import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, BertTokenizer
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup

text_model = "aubmindlab/bert-base-arabertv2"
# text_model = 'CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-egy'
print(f"Text Model: {text_model}")


class TextDataset(Dataset):
    def __init__(
        self,
        ids,
        text_data,
        labels,
        text_model,
        is_test=False,
    ):
        self.text_data = text_data
        self.ids = ids
        self.is_test = is_test
        # If not a test set, initialize labels
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_data[index]
        # if not self.is_test:
        label = self.labels[index]

        # tokenize text data
        text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=train_max_seq_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        fdata = {
            "id": id,
            "text": text["input_ids"].squeeze(0),
            "text_mask": text["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            fdata["label"] = torch.tensor(label, dtype=torch.long)
            return fdata
        else:
            return fdata


train_file = "arabic_memes_propaganda_araieval_24_train.json"
validation_file = "arabic_memes_propaganda_araieval_24_dev.json"
# test_file = 'arabic_memes_propaganda_araieval_24_test.json'

text_model_name = text_model

import json

import pandas as pd
import PIL
from tqdm import tqdm


def read_data(fpath, is_test=False):
    if is_test:
        data = {"id": [], "text": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in tqdm(js_obj):
            data["id"].append(obj["id"])
            data["text"].append(obj["text"])
    else:
        data = {"id": [], "text": [], "label": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in tqdm(js_obj):
            data["id"].append(obj["id"])
            data["text"].append(obj["text"])
            data["label"].append(obj["class_label"])
    return pd.DataFrame.from_dict(data)


l2id = {"not_propaganda": 0, "propaganda": 1}

train_df = read_data(train_file)
train_df["label"] = train_df["label"].map(l2id)
train_df = TextDataset(train_df["id"], train_df["text"], train_df["label"], text_model=text_model)

validation_df = read_data(validation_file)
validation_df["label"] = validation_df["label"].map(l2id)
validation_df = TextDataset(
    validation_df["id"], validation_df["text"], validation_df["label"], text_model=text_model
)

# test_df = read_data(test_file)
# #test_df['label'] = test_df['label'].map(l2id)
# test_df = MultimodalDataset(test_df['id'], test_df['text'], text_model=text_model) #, test_df['label']


if max_train_samples is not None:
    max_train_samples_n = min(len(train_df), max_train_samples)
    train_df = train_df.select(range(max_train_samples_n))


if max_eval_samples is not None:
    max_eval_samples_n = min(len(validation_df), max_eval_samples)
    validation_df = validation_df.select(range(max_eval_samples_n))


# if max_predict_samples is not None:
#     max_predict_samples_n = min(len(test_df), max_predict_samples)
#     predict_dataset = test_df.select(range(max_predict_samples_n))

import random

for index in random.sample(range(len(train_df)), 3):
    print(f"Sample {index} of the training set: {train_df[index]}.")

train_df = torch.utils.data.DataLoader(
    train_df, batch_size=8, shuffle=True, drop_last=True
)
validation_df = torch.utils.data.DataLoader(
    validation_df, batch_size=8, shuffle=True, drop_last=True
)

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from transformers import AutoModel, BertModel


# Define the multimodal classification model
# Define the multimodal classification model
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


class LLMWithClassificationHead(nn.Module):
    def __init__(
        self,
        model_name,
        pooling_type,
        num_classes,
        hidden_size=768,
        attention_hidden_size=512,
        cnn_kernel_size=3,
    ):
        super(LLMWithClassificationHead, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling_type = pooling_type
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        if pooling_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, attention_hidden_size),
                nn.Tanh(),
                nn.Linear(attention_hidden_size, 1),
            )
        elif pooling_type == "cnn":
            self.conv1d = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            )

        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
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

        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss, logits  # Modify return statement to include loss

        return logits  # Keep return statement for scenarios without labels

    def cls_pooling(self, outputs):
        return outputs.last_hidden_state[:, 0]

    def max_pooling(self, outputs):
        return torch.max(outputs.last_hidden_state, dim=1)[0]

    def mean_pooling(self, outputs, attention_mask):
        attention_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(outputs.last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(
            outputs.last_hidden_state * attention_mask_expanded, dim=1
        )
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def attention_pooling(self, outputs, attention_mask):
        attention_scores = self.attention(outputs.last_hidden_state)
        attention_scores = attention_scores.squeeze(-1)
        attention_scores = attention_scores + (1.0 - attention_mask) * -1e9
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_embeddings = torch.sum(
            outputs.last_hidden_state * attention_weights.unsqueeze(-1), dim=1
        )
        return weighted_embeddings

    def cnn_pooling(self, outputs):
        last_hidden_state = outputs.last_hidden_state.permute(0, 2, 1)
        cnn_outputs = self.conv1d(last_hidden_state)
        cnn_outputs = F.relu(cnn_outputs)
        pooled_output, _ = torch.max(cnn_outputs, dim=-1)
        return pooled_output


pooling_type = "attention"


# Define the training and testing functions
def train(
    model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler=None
):
    model.train()
    train_loss = 0.0
    correct = 0
    total_batches = len(train_loader)
    check_interval = total_batches // 10
    batch_losses = []

    for batch_idx, data in enumerate(train_loader, 1):
        optimizer.zero_grad()
        if USE_FP16:
            with autocast():
                text = data["text"].to(device)
                mask = data["text_mask"].to(device)
                labels = data["label"].to(device)
                try:
                    output = model(text, mask)
                except RuntimeError as e:
                    print("Error occurred during forward pass:")
                    print("Input data:", data)
                    print("Token IDs:", text)
                    print("Embedding weights shape:", model.model.embeddings.word_embeddings.weight.shape)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            max_grad_norm = 1.0  # Adjust the threshold as needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            text = data["text"].to(device)
            mask = data["text_mask"].to(device)
            labels = data["label"].to(device)
            try:
                output = model(text, mask)
            except RuntimeError as e:
                print("Error occurred during forward pass:")
                print("Input data:", data)
                print("Token IDs:", text)
                print("Embedding weights shape:", model.model.embeddings.word_embeddings.weight.shape)
            loss = criterion(output, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            max_grad_norm = 1.0  # Adjust the threshold as needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        train_loss += loss.item() * labels.size(0)
        batch_losses.append(loss.item())  # Append the loss for the current batch
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
            avg_loss = sum(batch_losses) / len(
                batch_losses
            )  # Calculate the average loss
            print(
                f"TRAIN | Epoch [{epoch}] | Batch [{batch_idx}/{total_batches}] | Loss: {avg_loss:.4f} | LR: {current_lr} | Grad Norm: {grad_norm:.4f} |"
            )
            batch_losses = []  # Reset the batch losses for the next 10 steps

        # Check test accuracy at equidistant intervals
        if batch_idx % check_interval == 0 or batch_idx == total_batches:
            test_loss, accuracy, macro_f1 = test(
                model, validation_df, criterion, device, epoch
            )
            print(
                f" TEST | Epoch [{epoch}] | Batch [{batch_idx}/{total_batches}] | Test Loss: {test_loss:.4f} | Acc: {accuracy:.4f} | F1: {macro_f1:.4f} |"
            )
            global best_macro_f1
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                evaluate(model, validation_df, device)

    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(
        f"TRAIN | Epoch [{epoch}] | Training Loss: {train_loss:.4f} | Accuracy: {accuracy:.4f} |"
    )
    return train_loss, accuracy


def test(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total_batches = len(test_loader)
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 1):
            if USE_FP16:
                with autocast():
                    text = data["text"].to(device)
                    mask = data["text_mask"].to(device)
                    labels = data["label"].to(device)
                    output = model(text, mask)
                    loss = criterion(output, labels)
            else:
                text = data["text"].to(device)
                mask = data["text_mask"].to(device)
                labels = data["label"].to(device)
                output = model(text, mask)
                loss = criterion(output, labels)

            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            if batch_idx % 10 == 0:
                print(
                    f" TEST | Epoch [{epoch}] | Batch [{batch_idx}/{total_batches}] | Loss: {loss.item():.4f} |"
                )

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    print(
        f" TEST | Epoch [{epoch}] | Testing Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f} | Macro F1: {macro_f1:.4f} |"
    )
    return test_loss, accuracy, macro_f1


def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    y_test_pred = []
    ids = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            text = data["text"].to(device)
            mask = data["text_mask"].to(device)
            output = model(text, mask)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted)
            ids.append(data["id"])

    team_name = "kevinmathew"
    fname = f"task2A_{team_name}.tsv"
    run_id = f"{team_name}_{text_model}_{pooling_type}.tsv"

    with open(fname, "w") as f:
        f.write("id\tlabel\trun_id\n")
        indx = 0
        id2l = {0: "not_propaganda", 1: "propaganda"}
        for i, line in enumerate(predictions):
            for indx, l in enumerate(line.tolist()):
                f.write(f"{ids[i][indx]}\t{id2l[l]}\t{run_id}\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LLMWithClassificationHead(
    num_classes=2, model_name=text_model, pooling_type=pooling_type
)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 5
total_steps = len(train_df) * num_epochs
warmup_steps = int(0.1 * total_steps)  # Adjust the warmup ratio as needed
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Train the model
for epoch in range(num_epochs):
    train_loss, acc = train(
        model, train_df, criterion, optimizer, scheduler, device, epoch, scaler
    )
    test_loss, accuracy, macro_f1 = test(model, validation_df, criterion, device, epoch)
    print(
        "  ALL | Epoch {}/{}: Train Loss = {:.4f}, Test Loss = {:.4f}, Train Accuracy = {:.4f}, Test Accuracy = {:.4f}, F1 = {:.4f}".format(
            epoch + 1, num_epochs, train_loss, test_loss, acc, accuracy, macro_f1
        )
    )
