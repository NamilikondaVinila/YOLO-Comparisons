import os
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
import csv
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve, auc, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
dataset_path = '/Users/martinprabhu/Downloads/uavdt'
config_path = '/Users/martinprabhu/Downloads/yolo/uavdt.yaml'
model_path = '/Users/martinprabhu/Downloads/yolo/yolov5.pt'
batch_size = 16
imgsz = 640
epochs = 500
project = 'uavdt_project'
name = 'yolov5m'
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# Load configuration file
with open(config_path, 'r') as f:
    uavdt_config = yaml.safe_load(f)

# Create YOLOv5 model
model = YOLO(model_path)
model.dataset = dataset_path
model.data = config_path

# Define custom dataset class
class UAVDTDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path)]
        self.labels = [int(img.split('_')[1].split('.')[0]) for img in os.listdir(dataset_path)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Define data augmentation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomCrop(size=imgsz, padding=10),
])

# Create dataset and data loader
dataset = UAVDTDataset(dataset_path, transform=transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Define loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Train the model
writer = SummaryWriter(log_dir='runs/' + project)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / len(val_dataset)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.3f}, Val Accuracy: {accuracy:.3f}')

# Evaluate the model
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Calculate confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{conf_mat}")

# Calculate precision, recall, F1 score, and accuracy
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"Accuracy: {accuracy:.3f}")

# Calculate classification report
report = classification_report(y_true, y_pred)
print(f"Classification Report:\n{report}")

# Calculate ROC AUC score
y_pred_proba = model.predict_proba(val_loader)
y_pred_proba = y_pred_proba[:, 1]
auc_score = roc_auc_score(y_true, y_pred_proba)
print(f"ROC AUC Score: {auc_score:.3f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Plot confusion matrix
sns.heatmap(conf_mat, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate error scores
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")

# Calculate inference time metrics
start_time = time.time()
model.eval()
with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
inference_time = time.time() - start_time
fps = 1 / inference_time
latency = inference_time
print(f"Inference Time: {inference_time:.3f} seconds")
print(f"Frames Per Second (FPS): {fps:.3f}")
print(f"Latency: {latency:.3f} seconds")

# Calculate true positives, false positives, false negatives, and true negatives
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")

# Calculate intersection over union (IoU)
iou = tp / (tp + fp + fn)
print(f"Intersection over Union (IoU): {iou:.3f}")

# Calculate mean average precision (mAP)
map_05 = 0
map_0595 = 0
for i in range(10):
    threshold = 0.5 + i * 0.05
    tp, fp, fn = 0, 0, 0
    for j in range(len(y_true)):
        if y_pred[j] >= threshold and y_true[j] == 1:
            tp += 1
        elif y_pred[j] >= threshold and y_true[j] == 0:
            fp += 1
        elif y_pred[j] < threshold and y_true[j] == 1:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    map_05 += precision
    map_0595 += 2 * precision * recall / (precision + recall)
map_05 /= 10
map_0595 /= 10
print(f"Mean Average Precision (mAP)@0.5: {map_05:.3f}")
print(f"Mean Average Precision (mAP)@0.5:0.95: {map_0595:.3f}")

# Save the model
torch.save(model.state_dict(), 'yolov5m.pth')

# Load the model
model.load_state_dict(torch.load('yolov5m.pth'))

# Evaluate the model on a single image
img_path = 'path/to/image.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.tensor(img).unsqueeze(0)
img = img.to(device)
outputs = model(img)
_, predicted = torch.max(outputs, 1)
print(f"Predicted class: {predicted.item()}")

# Visualize the output
plt.imshow(img.cpu().numpy().squeeze(0).transpose(1, 2, 0))
plt.title(f"Predicted class: {predicted.item()}")
plt.show()

# Save the output to a CSV file
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Predicted Class"])
    writer.writerow([img_path, predicted.item()])

# Create a tensorboard writer
writer = SummaryWriter(log_dir='runs/' + project)

# Add some metadata to the tensorboard writer
writer.add_text('Model', str(model))
writer.add_text('Dataset', str(dataset))
writer.add_text('Config', str(uavdt_config))

# Close the tensorboard writer
writer.close()

# Print some final messages
print("Training complete!")
print("Model saved to yolov5m.pth")
print("Output saved to output.csv")
print("Tensorboard logs saved to runs/" + project)