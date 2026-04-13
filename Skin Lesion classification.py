#importing modules
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
 

#setting up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using: {device}")
 

base      = r"/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
csv_file  = os.path.join(base, r"HAM10000_metadata.csv")
path1     = os.path.join(base, r"HAM10000_images_part_1")
path2     = os.path.join(base, r"HAM10000_images_part_2")
save_path = "/kaggle/working/resnet18_skin.pt"
ckpt_file = "/kaggle/working/best_model_params.pt"
 

dataset = pd.read_csv(csv_file)
 
map_label   = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
reverse_map = {v: k for k, v in map_label.items()}
 
dataset["label"] = dataset["dx"].map(map_label)
 
traindata, testdata = train_test_split(dataset, test_size=0.2, random_state=42)
traindata, valdata  = train_test_split(traindata, test_size=0.1, random_state=42)
 
print(f"train: {len(traindata)} | val: {len(valdata)} | test: {len(testdata)}")
 
#data trasnformation 
transform_train = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(20),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
transform_val = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
#custom dataset class so images cna be found from one path or the other
class HAM_dataset_loading:
    def __init__(self, data, path1, path2, transformation, maplabel):
        self.data           = data
        self.transformation = transformation
        self.maplabel       = maplabel
        self.imagepaths     = {}
 
        for imageid in data["image_id"]:
            p1 = os.path.join(path1, imageid + ".jpg")
            p2 = os.path.join(path2, imageid + ".jpg")
            if os.path.exists(p1):
                self.imagepaths[imageid] = p1
            elif os.path.exists(p2):
                self.imagepaths[imageid] = p2
            else:
                print(f"Image not found: {imageid}")
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        row      = self.data.iloc[index]
        image_id = row["image_id"]
        label    = torch.tensor(self.maplabel[row["dx"]], dtype=torch.long)
        image    = Image.open(self.imagepaths[image_id]).convert("RGB")
 
        if self.transformation:
            image = self.transformation(image)
 
        return image, label
 
 
#loading the data using dataloader
train_data = HAM_dataset_loading(traindata, path1, path2, transform_train, map_label)
val_data   = HAM_dataset_loading(valdata,   path1, path2, transform_val,   map_label)
test_data  = HAM_dataset_loading(testdata,  path1, path2, transform_val,   map_label)
 
batchsize  = 128
numworkers = 4
 
train_dataload = DataLoader(train_data, batch_size=batchsize, shuffle=True,  num_workers=numworkers, pin_memory=True)
val_dataload   = DataLoader(val_data,   batch_size=batchsize, shuffle=False, num_workers=numworkers, pin_memory=True)
test_dataload  = DataLoader(test_data,  batch_size=batchsize, shuffle=False, num_workers=numworkers, pin_memory=True)
 
 
#training of model
def training_model(model, criterion, optim, lr, data, epochnum, lengthofdataset):
    bestacc = 0
 
    for epoch in range(epochnum):
        print(f"epoch : {epoch + 1} / {epochnum}")
 
        for phase in ["train", "val"]:
            runningloss    = 0
            correctanswers = 0
 
            if phase == "train":
                model.train()
            else:
                model.eval()
 
            for images, labels in data[phase]:
                images = images.to(device)
                labels = labels.to(device)
 
                with torch.set_grad_enabled(phase == "train"):
                    output          = model(images)
                    _, predictions  = torch.max(output, 1)
                    loss            = criterion(output, labels)
 
                if phase == "train":
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
 
                runningloss    += loss.item() * images.shape[0]
                correctanswers += (predictions == labels).sum().item()
 
            acc       = correctanswers / lengthofdataset[phase] * 100
            epochloss = runningloss    / lengthofdataset[phase]
 
            print(f"correct: {correctanswers} / {lengthofdataset[phase]}  |  {phase} loss: {epochloss:.4f}  acc: {acc:.2f}%")
 
            if phase == "val" and acc > bestacc:
                bestacc = acc
                torch.save(model.state_dict(), ckpt_file)
                print(f"  saved new best model — val acc: {bestacc:.2f}%")
 
        lr.step()
 
    model.load_state_dict(torch.load(ckpt_file, weights_only=True))
    return model
 
 
#modifiying the resnet18 model
model         = models.resnet18(weights="IMAGENET1K_V1")
number_inputs = model.fc.in_features
model.fc      = nn.Linear(number_inputs, 7)
model         = model.to(device)
 
#imbalance weight handling
counts       = dataset["dx"].value_counts()
label_order  = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
actualcounts = [counts[label] for label in label_order]
 
total      = sum(actualcounts)
numclasses = len(label_order)
weight     = [total / (numclasses * count) for count in actualcounts]
weights_1  = torch.tensor(weight, dtype=torch.float32).to(device)
criterion  = nn.CrossEntropyLoss(weight=weights_1)
 
optim = torch.optim.Adam([
    {'params': model.fc.parameters(),
     'lr': 0.0001},
    {'params': [p for name, p in model.named_parameters() if 'fc' not in name],
     'lr': 0.00005}
])
 
lr = lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)
 
lengthdataset = {"train": len(train_data), "val": len(val_data)}
data          = {"train": train_dataload,   "val": val_dataload}
 
model = training_model(model, criterion, optim, lr, data, 20, lengthdataset)
 
 
#testing model on test data 
all_predictions = []
all_labels      = []
 
def testing_model(model, dataloader, label_order):
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predictions = torch.max(output, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    print(classification_report(all_labels, all_predictions, target_names=label_order))
 
testing_model(model, test_dataload, label_order)
 
 
#using gradcam to show pixels which contribute most to the classification
twenty_images_df     = testdata.sample(20, random_state=42).reset_index(drop=True)
twenty_images_labels = twenty_images_df["label"]
 
images_one = []
for image_id in twenty_images_df["image_id"]:
    p1 = os.path.join(path1, image_id + ".jpg")
    p2 = os.path.join(path2, image_id + ".jpg")
    img_path = p1 if os.path.exists(p1) else p2
    images_one.append(Image.open(img_path).convert("RGB"))
 
images = [transform_val(pil_img) for pil_img in images_one]
 
target_layers = [model.layer4[-1]]
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
model.eval()
 
for index, image in enumerate(images):
    single          = image.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_class = model(single).argmax(dim=1).item()
 
    targets = [ClassifierOutputTarget(twenty_images_labels.iloc[index])]
    heatmap = cam(input_tensor=single, targets=targets)
 
    orig_resized = images_one[index].resize((224, 224))
    img_np       = np.array(orig_resized).astype(np.float32) / 255.0
    overlay      = show_cam_on_image(img_np, heatmap[0], use_rgb=True)
 
    img_id     = twenty_images_df["image_id"].iloc[index]
    img_name   = os.path.basename(test_data.imagepaths.get(img_id, img_id + ".jpg"))
    true_label = reverse_map.get(twenty_images_labels.iloc[index], "unknown")
    pred_label = reverse_map.get(predicted_class, f"class_{predicted_class}")
 
    if index < 4:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"{img_name}", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM++", fontsize=9)
        axes[1].axis("off")
        fig.suptitle(
            f"ID: {img_id}  |  True: {true_label}  |  Predicted: {pred_label}",
            fontsize=11, fontweight="bold",
            color="green" if true_label == pred_label else "red"
        )
        plt.tight_layout()
        plt.show()
 
 
#saved the models weights
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
