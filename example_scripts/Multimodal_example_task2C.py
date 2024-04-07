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

learning_rate = 1e-5
train_max_seq_len = 512
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

text_model = "aubmindlab/bert-base-arabertv2"
# text_model = "distilbert-base-multilingual-cased"
# text_model = "FacebookAI/xlm-roberta-base"
# text_model = 'CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-egy'
english_text_model = "roberta-base"
image_model = "efficientnet_b5"
# image_model = "resnet50"
print(f"Image Model: {image_model} | Text Model: {text_model}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageCaptioning:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")
        self.model.eval()

    def generate_caption(self, images, texts):
        inputs = self.processor(images, texts, return_tensors="pt", padding=True).to("cuda", torch.float16)
        captions = self.model.generate(**inputs)
        captions = [self.processor.decode(capt, skip_special_tokens=True) for capt in captions]

        return captions

class MultimodalDataset(Dataset):
    def __init__(self, ids, text_data, image_data, labels, is_test=False):
        self.text_data = text_data
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
        # if not self.is_test:
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model
        )  # bert-base-multilingual-uncased
        self.english_tokenizer = AutoTokenizer.from_pretrained(
            english_text_model
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the image to 224x224
                transforms.RandomHorizontalFlip(),  # Apply horizontal flip randomly
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),  # Randomly change brightness, contrast, and saturation
                transforms.RandomRotation(
                    degrees=15
                ),  # Randomly rotate the image to a certain degree
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        # Initialize ImageCaptioning and precalculate captions
        self.image_cap = ImageCaptioning()
        self.precalculated_captions = self.precompute_captions()
        del self.image_cap

    def precompute_captions(self):
        conditional_gen_text = "a meme of"
        batch_size = 64  # Adjust based on your GPU memory and model size
        total_images = len(self.image_data)
        captions = []
        
        for start_idx in tqdm(range(0, total_images, batch_size)):
            end_idx = min(start_idx + batch_size, total_images)
            batch_images = [Image.open(self.image_data[i]).convert("RGB") for i in range(start_idx, end_idx)]
            batch_texts = [conditional_gen_text] * len(batch_images)
            
            with torch.no_grad():  # Ensure no gradients are computed to save memory
                batch_captions = self.image_cap.generate_caption(images=batch_images, texts=batch_texts)
            captions.extend(batch_captions)

        return captions


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_data[index]
        image = self.image_data[index]
        # if not self.is_test:
        label = self.labels[index]
        caption = self.precalculated_captions[index]
        image = self.transform(Image.open(image).convert("RGB"))
        # text += ' ' + caption

        # tokenize text data
        text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=train_max_seq_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        caption_text = self.english_tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=train_max_seq_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        fdata = {
            "id": id,
            "text": text["input_ids"].squeeze(0),
            "text_mask": text["attention_mask"].squeeze(0),
            "caption_text": caption_text["input_ids"].squeeze(0),
            "caption_text_mask": caption_text["attention_mask"].squeeze(0),
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

text_model_name = text_model
image_model_name = image_model

import json

import pandas as pd
import PIL

# from datasets import Image, Dataset,DatasetDict
from tqdm import tqdm

# Image.open(obj['img_path']).convert("RGB")


def read_data(fpath, is_test=False):
    if is_test:
        data = {"id": [], "text": [], "image": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in tqdm(js_obj):
            data["id"].append(obj["id"])
            data["image"].append(obj["img_path"])
            data["text"].append(obj["text"])
    else:
        data = {"id": [], "text": [], "image": [], "label": []}
        js_obj = json.load(open(fpath, encoding="utf-8"))
        for obj in tqdm(js_obj):
            data["id"].append(obj["id"])
            data["image"].append(obj["img_path"])
            data["text"].append(obj["text"])
            data["label"].append(obj["class_label"])
    return pd.DataFrame.from_dict(data)


l2id = {"not_propaganda": 0, "propaganda": 1}

from sklearn.utils.class_weight import compute_class_weight

train_df = read_data(train_file)
train_df["label"] = train_df["label"].map(l2id)
class_labels = train_df["label"].tolist()
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"class weights: {class_weights}")
train_df = MultimodalDataset(
    train_df["id"], train_df["text"], train_df["image"], train_df["label"]
)

validation_df = read_data(validation_file)
validation_df["label"] = validation_df["label"].map(l2id)
validation_df = MultimodalDataset(
    validation_df["id"],
    validation_df["text"],
    validation_df["image"],
    validation_df["label"],
)

print("validation_df len:", len(validation_df))

# test_df = read_data(test_file)
# #test_df['label'] = test_df['label'].map(l2id)
# test_df = MultimodalDataset(test_df['id'], test_df['text'], test_df['image']) #, test_df['label']


# if max_train_samples is not None:
#     max_train_samples_n = min(len(train_df), max_train_samples)
#     train_df = train_df.select(range(max_train_samples_n))


# if max_eval_samples is not None:
#     max_eval_samples_n = min(len(validation_df), max_eval_samples)
#     validation_df = validation_df.select(range(max_eval_samples_n))


# if max_predict_samples is not None:
#     max_predict_samples_n = min(len(test_df), max_predict_samples)
#     predict_dataset = test_df.select(range(max_predict_samples_n))

import random

# for index in random.sample(range(len(train_df)), 3):
#     print(f"Sample {index} of the training set: {train_df[index]}.")

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
        hidden_size=768,
        attention_hidden_size=512,
        cnn_kernel_size=3,
    ):
        super(LLMWithClassificationHead, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling_type = pooling_type
        self.hidden_size = hidden_size

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

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling_type == "cls":
            pooled_output = self.cls_pooling(outputs)
        if self.pooling_type == "nopooling":
            pooled_output = self.last_hidden_state(outputs)
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

        return pooled_output  # Keep return statement for scenarios without labels

    def last_hidden_state(self, outputs):
        return outputs.last_hidden_state
    
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


pooling_type = "nopooling"

class MCA(nn.Module):
    def __init__(self, units):
        super(MCA, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)
        self.reduce = nn.Linear(2 * units, units)

    def forward(self, text_features, image_features):
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        image_features_with_time_axis = image_features.unsqueeze(1)

        score = torch.tanh(self.W1(text_features) + self.W2(image_features_with_time_axis))

        attention_weights = F.softmax(self.V(score), dim=1)

        context_vector1 = attention_weights * text_features
        # context_vector2 = attention_weights * image_features_with_time_axis

        context_vector1 = torch.sum(context_vector1, dim=1)
        # context_vector2 = torch.sum(context_vector2, dim=1)
        # context_vector = torch.cat([context_vector1, context_vector2], dim=1)

        # context_vector = self.reduce(context_vector)

        return context_vector1
    

class MCA3(nn.Module):
    def __init__(self, units):
        super(MCA3, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.W3 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)
        self.reduce = nn.Linear(2 * units, units)

    def forward(self, text_features, image_features, caption_features):
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        image_features_with_time_axis = image_features.unsqueeze(1)

        score = torch.tanh(self.W1(text_features) + self.W2(image_features_with_time_axis) + self.W3(caption_features))

        attention_weights = F.softmax(self.V(score), dim=1)

        context_vector1 = attention_weights * text_features
        context_vector2 = attention_weights * caption_features

        context_vector1 = torch.sum(context_vector1, dim=1)
        context_vector2 = torch.sum(context_vector2, dim=1)
        context_vector = torch.cat([context_vector1, context_vector2], dim=1)

        context_vector = self.reduce(context_vector)

        return context_vector

class ConcatAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(ConcatAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )

        self.reduce = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.BatchNorm1d(attention_dim),
            nn.ReLU(),
        )

    def forward(self, text_features, image_features):
        concatenated_features = torch.cat((text_features, image_features), dim=1)
        attention_weights = self.attention_layer(concatenated_features)
        attended_features = attention_weights * concatenated_features
        attended_features = self.reduce(attended_features)
        # print(f"Sizes: {concatenated_features.size()} | {attention_weights.size()} | {attended_features.size()} | {attended_features.sum(dim=1).size()} |")
        return attended_features


class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super(CrossModalAttention, self).__init__()
        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads
        )
        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads
        )
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, text_features, image_features):
        # Reshape features to have batch dimension
        text_features = text_features.unsqueeze(
            0
        )  # Shape: (batch_size, 1, feature_dim)
        image_features = image_features.unsqueeze(
            0
        )  # Shape: (batch_size, 1, feature_dim)

        # Apply cross-attention from text to image
        attended_image_features, _ = self.text_to_image_attention(
            query=text_features, key=image_features, value=image_features
        )

        # Apply cross-attention from image to text
        attended_text_features, _ = self.image_to_text_attention(
            query=image_features, key=text_features, value=text_features
        )

        # Combine attended features
        combined_features = (
            attended_text_features.sum(dim=0) + attended_image_features.sum(dim=0)
        ) / 2
        combined_features = self.bn(combined_features)

        return combined_features


class SelfAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super(SelfAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads
        )
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, text_features, image_features):
        # Concatenate features from both modalities
        features = torch.cat(
            (text_features.unsqueeze(0), image_features.unsqueeze(0)), dim=0
        )
        # Apply multi-head attention
        attended_features, _ = self.attention(features, features, features)
        # You might want to combine or process these features further
        combined_features = attended_features.sum(dim=0)
        combined_features = self.bn(combined_features)
        return combined_features


fusion_method = "mca"  # ['mca', 'concatenation', 'cross_modal', 'self_attention']
print(f"Using Fusion: {fusion_method}")


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, fusion_method):
        super(MultimodalClassifier, self).__init__()

        # Initialize text model from a pre-trained model
        self.text_model = LLMWithClassificationHead(
            model_name=text_model, pooling_type=pooling_type
        )
        self.text_dropout = nn.Dropout(0.3)
        text_hidden_size = 768

        # Fully connected layer for text features
        self.text_fc = nn.Sequential(
            nn.Linear(text_hidden_size, 512), nn.BatchNorm1d(512), nn.ReLU()
        )

        self.caption_text_model = LLMWithClassificationHead(
            model_name=english_text_model, pooling_type=pooling_type
        )
        self.caption_text_dropout = nn.Dropout(0.3)
        caption_text_hidden_size = 768

        # Fully connected layer for text features
        self.caption_text_fc = nn.Sequential(
            nn.Linear(caption_text_hidden_size, 512), nn.BatchNorm1d(512), nn.ReLU()
        )

        # Initialize image model from a pre-trained model
        self.image_model = timm.create_model(image_model_name, pretrained=True)
        print(f"in features before: {self.image_model.classifier.in_features}")
        self.image_model.classifier = nn.Sequential(
            nn.Linear(self.image_model.classifier.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.fusion_method = fusion_method
        if fusion_method == "concatenation":
            self.fusion_layer = ConcatAttention(1024, 512)
        elif fusion_method == "mca":
            self.fusion_layer = MCA3(512)
        elif fusion_method == "cross_modal":
            self.fusion_layer = CrossModalAttention(512)
        elif fusion_method == "self_attention":
            self.fusion_layer = SelfAttentionFusion(512)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        fusion_output_size = (
            512
            if fusion_method in ["concatenation", "cross_modal", "self_attention"]
            else 512
        )
        self.output_fc = nn.Sequential(
            nn.Linear(fusion_output_size, num_classes), nn.BatchNorm1d(num_classes)
        )

    def get_params(self, lr):
        attention_params = []
        text_model_params = []
        image_model_params = []

        for name, param in self.named_parameters():
            if "fusion_layer" in name:
                attention_params.append(param)
            elif "text_model" in name:
                text_model_params.append(param)
            elif "image_model" in name:
                image_model_params.append(param)
            else:
                attention_params.append(param)

        return [
            {"params": attention_params, "lr": lr},
            {"params": text_model_params, "lr": lr * 0.8},
            {"params": image_model_params, "lr": lr * 0.8},
        ]

    def forward(self, text, image, mask, caption_text, caption_text_mask):
        text_output = self.text_model(text, attention_mask=mask)
        text_output = self.text_dropout(text_output)
        text_output = self.text_fc(text_output)

        caption_text_output = self.caption_text_model(caption_text, attention_mask=caption_text_mask)
        caption_text_output = self.caption_text_dropout(caption_text_output)
        caption_text_output = self.caption_text_fc(caption_text_output)

        image_output = self.image_model(image)

        if hasattr(self, "fusion_layer"):
            fused_output = self.fusion_layer(text_output, image_output, caption_text_output)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        output = self.output_fc(fused_output)

        return output


# Define the training and testing functions
def train(
    model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler=None
):
    model.train()
    train_loss = 0.0
    correct = 0
    total_batches = len(train_loader)
    check_interval = total_batches // 2
    batch_losses = []

    for batch_idx, data in enumerate(train_loader, 1):
        optimizer.zero_grad()
        if USE_FP16:
            with autocast():
                image = data["image"].to(device)
                text = data["text"].to(device)
                mask = data["text_mask"].to(device)
                caption_text = data["caption_text"].to(device)
                caption_text_mask = data["caption_text_mask"].to(device)
                labels = data["label"].to(device)
                output = model(text, image, mask, caption_text, caption_text_mask)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            max_grad_norm = 1.0  # Adjust the threshold as needed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            image = data["image"].to(device)
            text = data["text"].to(device)
            mask = data["text_mask"].to(device)
            caption_text = data["caption_text"].to(device)
            caption_text_mask = data["caption_text_mask"].to(device)
            labels = data["label"].to(device)
            output = model(text, image, mask, caption_text, caption_text_mask)
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
                    text = data["text"].to(device)
                    mask = data["text_mask"].to(device)
                    caption_text = data["caption_text"].to(device)
                    caption_text_mask = data["caption_text_mask"].to(device)
                    labels = data["label"].to(device)
                    output = model(text, image, mask, caption_text, caption_text_mask)
                    loss = criterion(output, labels)
            else:
                image = data["image"].to(device)
                text = data["text"].to(device)
                mask = data["text_mask"].to(device)
                caption_text = data["caption_text"].to(device)
                caption_text_mask = data["caption_text_mask"].to(device)
                labels = data["label"].to(device)
                output = model(text, image, mask, caption_text, caption_text_mask)
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
            text = data["text"].to(device)
            mask = data["text_mask"].to(device)
            caption_text = data["caption_text"].to(device)
            caption_text_mask = data["caption_text_mask"].to(device)
            output = model(text, image, mask, caption_text, caption_text_mask)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted)
            ids.append(data["id"])

    team_name = "kevinmathew"
    fname = f"task2C_{team_name}.tsv"
    run_id = f"{team_name}_{image_model}_{text_model}_{english_text_model}_{fusion_method}.tsv"

    with open(fname, "w") as f:
        f.write("id\tlabel\trun_id\n")
        indx = 0
        id2l = {0: "not_propaganda", 1: "propaganda"}
        for i, line in enumerate(predictions):
            for indx, l in enumerate(line.tolist()):
                f.write(f"{ids[i][indx]}\t{id2l[l]}\t{run_id}\n")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalClassifier(num_classes=2, fusion_method=fusion_method)
model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.get_params(learning_rate))
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
