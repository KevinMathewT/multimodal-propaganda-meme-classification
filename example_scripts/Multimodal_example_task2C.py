# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_train.json
# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_dev.json
# !wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_araieval_24_train_dev.tar.gz

# !tar -xvzf arabic_memes_araieval_24_train_dev.tar.gz

# !pip install transformers
# !pip install datasets
# !pip install evaluate
# !pip install --upgrade accelerate

learning_rate=2e-5
num_train_epochs=2
train_max_seq_len = 512
max_train_samples = None
max_eval_samples=None
max_predict_samples=None
batch_size = 16

import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, BertTokenizer

text_model = 'aubmindlab/bert-base-arabertv2'
image_model = 'efficientnet_b4'

class MultimodalDataset(Dataset):
    def __init__(self, ids, text_data, image_data, labels, is_test=False):
        self.text_data = text_data
        self.image_data = image_data
        self.ids = ids
        self.is_test = is_test
        #if not self.is_test:
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(text_model) #bert-base-multilingual-uncased
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                             ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.ids[index]
        text = self.text_data[index]
        image = self.image_data[index]
        #if not self.is_test:
        label = self.labels[index]

        # tokenize text data
        text = self.tokenizer.encode_plus(text, add_special_tokens=True,
                                           max_length=train_max_seq_len, padding='max_length',
                                           return_attention_mask=True, return_tensors='pt')

        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = self.transform(Image.open(image).convert("RGB"))

        fdata = {
            'id': id,
            'text': text['input_ids'].squeeze(0),
            'text_mask': text['attention_mask'].squeeze(0),
            'image': image,
        }
        if not self.is_test:
            fdata['label'] = torch.tensor(label, dtype=torch.long)
            return fdata
        else:
            return fdata


train_file = 'arabic_memes_propaganda_araieval_24_train.json'
validation_file = 'arabic_memes_propaganda_araieval_24_dev.json'
# test_file = 'arabic_memes_propaganda_araieval_24_test.json'

text_model_name = text_model
image_model_name = image_model

import json

import pandas as pd
import PIL
#from datasets import Image, Dataset,DatasetDict
from tqdm import tqdm

# Image.open(obj['img_path']).convert("RGB")

def read_data(fpath, is_test=False):
  if is_test:
    data = {'id': [], 'text': [], 'image': []}
    js_obj = json.load(open(fpath, encoding='utf-8'))
    for obj in tqdm(js_obj):
      data['id'].append(obj['id'])
      data['image'].append(obj['img_path'])
      data['text'].append(obj['text'])
  else:
    data = {'id': [], 'text': [], 'image': [], 'label': []}
    js_obj = json.load(open(fpath, encoding='utf-8'))
    for obj in tqdm(js_obj):
      data['id'].append(obj['id'])
      data['image'].append(obj['img_path'])
      data['text'].append(obj['text'])
      data['label'].append(obj['class_label'])
  return pd.DataFrame.from_dict(data)


l2id = {'not_propaganda': 0, 'propaganda': 1}

train_df = read_data(train_file)
train_df['label'] = train_df['label'].map(l2id)
train_df = MultimodalDataset(train_df['id'], train_df['text'], train_df['image'], train_df['label'])

validation_df = read_data(validation_file)
validation_df['label'] = validation_df['label'].map(l2id)
validation_df = MultimodalDataset(validation_df['id'], validation_df['text'], validation_df['image'], validation_df['label'])

# test_df = read_data(test_file)
# #test_df['label'] = test_df['label'].map(l2id)
# test_df = MultimodalDataset(test_df['id'], test_df['text'], test_df['image']) #, test_df['label']


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

train_df = torch.utils.data.DataLoader(train_df, batch_size=8, shuffle=True, drop_last=True)
validation_df = torch.utils.data.DataLoader(validation_df, batch_size=8, shuffle=True, drop_last=True)

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from transformers import AutoModel, BertModel


# Define the multimodal classification model
# Define the multimodal classification model
import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, fusion_method):
        super(MultimodalClassifier, self).__init__()
        
        # Initialize text model from a pre-trained model
        self.text_model = AutoModel.from_pretrained(text_model_name)  # [batch_size, sequence_length, hidden_size]
        self.text_dropout = nn.Dropout(0.3)
        text_hidden_size = self.text_model.config.hidden_size
        
        # Fully connected layer for text features
        self.text_fc = nn.Linear(text_hidden_size, 512)  # [batch_size, 512]
        
        # Initialize image model from a pre-trained model
        self.image_model = timm.create_model(image_model_name, pretrained=True)  # [batch_size, num_features, height, width]
        image_hidden_size = self.image_model.num_features
        
        # Pooling layer to reduce image dimensions
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))  # [batch_size, num_features, 1, 1]
        self.image_fc = nn.Linear(image_hidden_size, 512)  # [batch_size, 512]
        
        self.fusion_method = fusion_method
        # Dynamically adjust the input size of the output layer based on the fusion method
        fusion_output_size = 1024 if fusion_method in ['concatenation', 'attention_fusion', 'bilinear_fusion', 'gated_fusion'] else 512
        self.output_fc = nn.Linear(fusion_output_size, num_classes)  # [batch_size, num_classes]
    
    def forward(self, text, image, mask):
        # Text processing
        text_output = self.text_model(text, attention_mask=mask).last_hidden_state
        text_output = self.text_dropout(text_output[:, 0, :])
        text_output = self.text_fc(text_output)  # [batch_size, 512]
        
        # Image processing
        image_output = self.image_model(image)
        print(f"image_output dim: {image_output.size()}")
        image_output = self.image_pool(image_output).flatten(1)
        image_output = self.image_fc(image_output)  # [batch_size, 512]
        
        # Fusion
        if hasattr(self, self.fusion_method):
            fused_output = getattr(self, self.fusion_method)(text_output, image_output)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        
        # Final classification layer
        output = self.output_fc(fused_output)
        
        return output

    # Fusion methods
    def concatenation(text_features, image_features):
        return torch.cat((text_features, image_features), dim=1)  # Output: (batch_size, 1024)

    def addition(text_features, image_features):
        return text_features + image_features  # Output: (batch_size, 512)

    def subtraction(text_features, image_features):
        return text_features - image_features  # Output: (batch_size, 512)

    def multiplication(text_features, image_features):
        return text_features * image_features  # Output: (batch_size, 512)

    def attention_fusion(text_features, image_features):
        # Compute attention weights
        attention_weights = torch.matmul(text_features, image_features.transpose(1, 2))  # Output: (batch_size, 512, 512)
        attention_weights = torch.softmax(attention_weights, dim=2)  # Output: (batch_size, 512, 512)
        
        # Attend to image features
        attended_image_features = torch.matmul(attention_weights, image_features)  # Output: (batch_size, 512, 512)
        
        # Concatenate attended image features with text features
        fused_features = torch.cat((text_features, attended_image_features), dim=1)  # Output: (batch_size, 1024)
        
        return fused_features

    def bilinear_fusion(text_features, image_features):
        # Compute bilinear interaction
        bilinear_interaction = torch.matmul(text_features, image_features.transpose(1, 2))  # Output: (batch_size, 512, 512)
        bilinear_interaction = torch.flatten(bilinear_interaction, start_dim=1)  # Output: (batch_size, 512*512)
        
        return bilinear_interaction

    def gated_fusion(text_features, image_features):
        # Compute gate weights
        gate_weights = torch.sigmoid(torch.matmul(text_features, image_features.transpose(1, 2)))  # Output: (batch_size, 512, 512)
        
        # Apply gate to image features
        gated_image_features = gate_weights * image_features  # Output: (batch_size, 512, 512)
        
        # Concatenate gated image features with text features
        fused_features = torch.cat((text_features, gated_image_features), dim=1)  # Output: (batch_size, 1024)
        
        return fused_features

# Define the training and testing functions
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0.0
    correct = 0
    total_batches = len(train_loader)
    for batch_idx, data in enumerate(train_loader, 1):
        optimizer.zero_grad()
        text = data["text"].to(device)
        image = data["image"].to(device)
        mask = data["text_mask"].to(device)
        labels = data['label'].to(device)
        print(f"image dim: {image.size()}")
        output = model(text, image, mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"| Epoch [{epoch}] | Batch [{batch_idx}/{total_batches}] | Loss: {loss.item():.4f} |")
    
    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print(f"| Epoch [{epoch}] | Training Loss: {train_loss:.4f} | Accuracy: {accuracy:.4f} |")
    return train_loss, accuracy

def test(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0.0
    correct = 0
    total_batches = len(test_loader)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 1):
            text = data["text"].to(device)
            image = data["image"].to(device)
            mask = data["text_mask"].to(device)
            labels = data['label'].to(device)
            output = model(text, image, mask)
            loss = criterion(output, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"| Epoch [{epoch}] | Batch [{batch_idx}/{total_batches}] | Loss: {loss.item():.4f} |")
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"| Epoch [{epoch}] | Testing Loss: {test_loss:.4f} | Accuracy: {accuracy:.4f} |")
    return test_loss, accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalClassifier(num_classes=2, fusion_method='attention_fusion')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, acc = train(model, train_df, criterion, optimizer, device, epoch)
    dev_loss, accuracy = test(model, validation_df, criterion, device, epoch)
    print('Epoch {}/{}: Train Loss = {:.4f}, Test Loss = {:.4f}, Train Accuracy = {:.4f}, Test Accuracy = {:.4f}'.format(epoch+1, num_epochs, train_loss, dev_loss, acc, accuracy))


def evaluate(model, test_loader, device):
    model.eval()
    predictions = []
    y_test_pred = []
    ids = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            text = data["text"].to(device)
            image = data["image"].to(device)
            mask = data["text_mask"].to(device)
            output = model(text, image, mask)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted)
            ids.append(data["id"])

    team_name = "kevinmathew"
    fname = f'task2C_{team_name}.tsv'

    with open(fname, 'w') as f:
      f.write("id\tlabel\trun_id\n")
      indx = 0
      id2l = {0:'not_propaganda', 1:'propaganda'}
      for i, line in enumerate(predictions):
        for indx, l in enumerate(line.tolist()):
          f.write(f"{ids[i][indx]}\t{id2l[l]}\tDistilBERT+ResNet\n")

evaluate(model, validation_df, device)


