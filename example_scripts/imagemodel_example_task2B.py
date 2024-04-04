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

learning_rate = 5e-5
max_train_samples = None
max_eval_samples = None
max_predict_samples = None
batch_size = 16
best_macro_f1 = 0.0

import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, BertTokenizer
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup


class MultimodalDataset(Dataset):
    def __init__(self, ids, image_data, labels, is_test=False):
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
        # if not self.is_test:
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),  # Resize the image to 224x224
                # transforms.RandomHorizontalFlip(),  # Apply horizontal flip randomly
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),  # Randomly change brightness, contrast, and saturation
                # transforms.RandomRotation(
                #     degrees=15
                # ),  # Randomly rotate the image to a certain degree
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.ids[index]
        image = self.image_data[index]
        # if not self.is_test:
        label = self.labels[index]

        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = self.transform(Image.open(image).convert("RGB"))

        fdata = {
            "id": id,
            "image": image,
        }
        if not self.is_test:
            fdata["label"] = torch.tensor(label, dtype=torch.long)
            return fdata
        else:
            return fdata


train_file = "arabic_memes_propaganda_araieval_24_train.json"
validation_file = "arabic_memes_propaganda_araieval_24_dev.json"
# test_file = 'arabic_memes_propaganda_araieval_24_test.json'

import json

import pandas as pd
import PIL

# from datasets import Image, Dataset,DatasetDict
from tqdm import tqdm

# Image.open(obj['img_path']).convert("RGB")


def read_data(fpath, is_test=False):
    if is_test:
        data = {"id": [], "image": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in tqdm(js_obj):
            data["id"].append(obj["id"])
            data["image"].append(obj["img_path"])
    else:
        data = {"id": [], "image": [], "label": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in tqdm(js_obj):
            data["id"].append(obj["id"])
            data["image"].append(obj["img_path"])
            data["label"].append(obj["class_label"])
    return pd.DataFrame.from_dict(data)


l2id = {"not_propaganda": 0, "propaganda": 1}

train_df = read_data(train_file)
train_df["label"] = train_df["label"].map(l2id)
train_df = MultimodalDataset(train_df["id"], train_df["image"], train_df["label"])

validation_df = read_data(validation_file)
validation_df["label"] = validation_df["label"].map(l2id)
validation_df = MultimodalDataset(
    validation_df["id"],
    validation_df["image"],
    validation_df["label"],
)

print("validation_df len:", len(validation_df))

# test_df = read_data(test_file)
# #test_df['label'] = test_df['label'].map(l2id)
# test_df = MultimodalDataset(test_df['id'], test_df['image']) #, test_df['label']


# if max_train_samples is not None:
#     max_train_samples_n = min(len(train_df), max_train_samples)
#     train_df = train_df.select(range(max_train_samples_n))


# if max_eval_samples is not None:
#     max_eval_samples_n = min(len(validation_df), max_eval_samples)
#     validation_df = validation_df.select(range(max_eval_samples_n))


# if max_predict_samples is not None:
#     max_predict_samples_n = min(len(test_df), max_predict_samples)
#     predict_dataset = test_df.select(range(max_predict_samples_n))

train_df = torch.utils.data.DataLoader(
    train_df, batch_size=batch_size, shuffle=True, drop_last=True
)
validation_df = torch.utils.data.DataLoader(
    validation_df, batch_size=batch_size, shuffle=True, drop_last=False
)

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from transformers import AutoModel, BertModel


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=2, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class SEResNeXt50_32x4d_BH(nn.Module):
    name = "SEResNeXt50_32x4d_BH"

    def __init__(self, pretrained=True):
        super().__init__()
        self.model_arch = "seresnext50_32x4d"
        self.net = nn.Sequential(
            *list(timm.create_model(self.model_arch, pretrained=pretrained).children())[
                :-2
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(2, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        img_feature = self.net(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output


class ResNeXt50_32x4d_BH(nn.Module):
    name = "ResNeXt50_32x4d_BH"

    def __init__(self, pretrained=True):
        super().__init__()
        self.model_arch = "resnext50_32x4d"
        self.model = timm.create_model(self.model_arch, pretrained=pretrained)
        model_list = list(self.model.children())
        model_list[-1] = nn.Identity()
        model_list[-2] = nn.Identity()
        self.net = nn.Sequential(*model_list)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(2, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x):
        x = self.net(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fea_bn(x)
        # fea = self.dropout(fea)
        x = self.binary_head(x)
        # x = self.fc(x)

        return x


class ViTBase16_BH(nn.Module):
    name = "ViTBase16_BH"

    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model("vit_base_patch16_384", pretrained=pretrained)
        self.net.norm = nn.Identity()
        self.net.head = nn.Identity()
        self.fea_bn = nn.BatchNorm1d(768)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(2, emb_size=768, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.net(x)
        x = self.fea_bn(x)
        # fea = self.dropout(fea)
        x = self.binary_head(x)
        return x


class ViTBase16(nn.Module):
    name = "ViTBase16"

    def __init__(self, pretrained=True):
        super().__init__()
        # self.model_arch = 'ViT-B_16'
        # self.net = VisionTransformer.from_pretrained(
        #     self.model_arch, num_classes=5) if pretrained else VisionTransformer.from_name(self.model_arch, num_classes=5)
        # print(self.model)

        self.model_arch = "vit_base_patch16_384"
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.head.in_features
        self.net.head = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.net(x)
        return x


class ViTLarge16(nn.Module):
    name = "ViTLarge16"

    def __init__(self, pretrained=True):
        super().__init__()
        # self.model_arch = 'ViT-B_16'
        # self.net = VisionTransformer.from_pretrained(
        #     self.model_arch, num_classes=5) if pretrained else VisionTransformer.from_name(self.model_arch, num_classes=5)
        # print(self.model)

        self.model_arch = "vit_large_patch16_384"
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.head.in_features
        self.net.head = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientNetB4(nn.Module):
    name = "EfficientNetB4"

    def __init__(self, pretrained=True):
        super().__init__()
        self.model_arch = "tf_efficientnet_b4_ns"
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.net(x)
        return x


class EfficientNetB3(nn.Module):
    name = "EfficientNetB3"

    def __init__(self, pretrained=True):
        super().__init__()
        self.model_arch = "tf_efficientnet_b3_ns"
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.net(x)
        return x
    

class EfficientNetB(nn.Module):

    def __init__(self, b, pretrained=True):
        super().__init__()
        name = f"EfficientNetB{b}"
        self.model_arch = f"tf_efficientnet_b{b}_ns"
        self.net = timm.create_model(self.model_arch, pretrained=pretrained)
        n_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(n_features, 2)

    def forward(self, x):
        x = self.net(x)
        return x


class GeneralizedMemesClassifier(nn.Module):
    def __init__(self, model_arch, n_class=2, pretrained=True):
        super().__init__()
        self.name = model_arch
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        model_list = list(self.model.children())
        model_list[-1] = nn.Linear(
            in_features=model_list[-1].in_features, out_features=n_class, bias=True
        )
        self.model = nn.Sequential(*model_list)

    def forward(self, x):
        x = self.model(x)
        return x


nets = {
    "SEResNeXt50_32x4d_BH": SEResNeXt50_32x4d_BH,
    "ViTBase16_BH": ViTBase16_BH,
    "ResNeXt50_32x4d_BH": ResNeXt50_32x4d_BH,
    "ViTBase16": ViTBase16,
    "ViTLarge16": ViTLarge16,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB": EfficientNetB,
}


image_model = "EfficientNetB"
# image_model = "resnet50"
print(f"Image Model: {image_model}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nets[image_model](b=0)
model.to(device)


# Define the training and testing functions
def train(
    model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler=None
):
    model.train()
    train_loss = 0.0
    correct = 0
    total_batches = len(train_loader)
    check_interval = total_batches // 1
    batch_losses = []

    for batch_idx, data in enumerate(train_loader, 1):
        optimizer.zero_grad()
        if USE_FP16:
            with autocast():
                image = data["image"].to(device)
                labels = data["label"].to(device)
                output = model(image)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            max_grad_norm = 1.0  # Adjust the threshold as needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            image = data["image"].to(device)
            labels = data["label"].to(device)
            output = model(image)
            loss = criterion(output, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            max_grad_norm = 10.0  # Adjust the threshold as needed
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
                    image = data["image"].to(device)
                    labels = data["label"].to(device)
                    output = model(image)
                    loss = criterion(output, labels)
            else:
                image = data["image"].to(device)
                labels = data["label"].to(device)
                output = model(image)
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
            image = data["image"].to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted)
            ids.append(data["id"])

    team_name = "kevinmathew"
    fname = f"task2B_{team_name}.tsv"
    run_id = f"{team_name}_{image_model}_binary_head.tsv"

    with open(fname, "w") as f:
        f.write("id\tlabel\trun_id\n")
        indx = 0
        id2l = {0: "not_propaganda", 1: "propaganda"}
        for i, line in enumerate(predictions):
            for indx, l in enumerate(line.tolist()):
                f.write(f"{ids[i][indx]}\t{id2l[l]}\t{run_id}\n")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=False
)
num_epochs = 20
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
