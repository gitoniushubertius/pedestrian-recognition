import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import numpy as np
from torchvision.ops import box_iou
import cv2
from torch.optim import Adam
from dataset import CaltechPedestrianDataset
import tqdm
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
import csv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

def get_model(device, name, path):
    if name == 'ssd':
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        model.to(device)
    elif name == 'vgg':
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        model.to(device)

    if path:
        model.load_state_dict(torch.load(path))
    return model

def get_best_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
    return device

def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, 0)
    return images, targets

def evaluate(model, val_data_loader, device):
    map_metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.4])
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        with tqdm.tqdm(val_data_loader, unit="batch") as pbar:
            for images, targets in pbar:
                images = [img.to(device) for img in images]
                
                outputs = model(images)

                preds = []
                for output in outputs:
                    mask = output["labels"] == 1 
                    preds.append({
                        "boxes": output["boxes"][mask].detach().cpu(),
                        "scores": output["scores"][mask].detach().cpu(),
                        "labels": output["labels"][mask].detach().cpu()
                    })

                gt = []
                for target in targets:
                    gt.append({
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu()
                    })

                all_preds.extend(preds)
                all_targets.extend(gt)

        map_result = map_metric(all_preds, all_targets)
        mAP50 = map_result["map_50"].item() 
        mAP = map_result["map"].item() 

        return mAP50, mAP


def train(model, model_name, _epochs=30, _batch_size=8, _lr=0.0001, device='cpu'):

    train_losses = []
    mAP50s = []
    mAPs = []
   
    dataset = CaltechPedestrianDataset(root_dir='dataset', split='train')
    data_loader = DataLoader(dataset, batch_size=_batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = CaltechPedestrianDataset(
        root_dir='dataset',
        split="val"
    )
    val_data_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = Adam(model.parameters(), lr=_lr)

    best_map = 0
    for epoch in range(_epochs):
        train_loss = 0
        model.train()
        with tqdm.tqdm(data_loader, unit="batch") as pbar:
            for images, targets in pbar:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                images = list(image for image in images)

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                optimizer.step()

                train_loss += losses.item()
        
        train_losses.append(train_loss / len(data_loader))
        
        mAP50, mAP = evaluate(model, val_data_loader, device)

        mAP50s.append(mAP50)
        mAPs.append(mAP)

        if best_map < mAP:
            best_map = mAP
            model_save_path = f'{model_name}_{_epochs}_{_batch_size}_{_lr}.pth'
            torch.save(model.state_dict(), model_save_path)
            print("Model saved successfully!")

        model_checkpoint_path = f'{model_name}_{_epochs}_{_batch_size}_{_lr}_checkpoint.pth'
        torch.save(model.state_dict(), model_checkpoint_path)
        print("Model saved successfully!")

        print(f"Epoch {epoch}, Loss: {train_loss / len(data_loader)}")
        print(f"mAP50: {mAP:.4f}")


    metrics_path = f'training_metrics_{model_name}_{_epochs}_{_batch_size}_{_lr}.csv'
    with open(metrics_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Epoch', 'Train Loss', 'Map50', 'Map'])

        for epoch in range(_epochs):
            writer.writerow([epoch+1, train_losses[epoch], mAP50s[epoch], mAPs[epoch]])

    return model

def draw_boxes(images, boxes, labels, scores, true_boxes=None, true_labels=None, threshold=0.5):
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    axes = ax.flatten()
    
    for i, ax in enumerate(axes):
        if i >= len(images):
            ax.axis("off")
            continue

        image = images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1) 

        ax.imshow(image)
        ax.axis("off")

        if true_boxes and true_labels:
            for box, label in zip(true_boxes[i], true_labels[i]):
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"GT: {label}", color='green', fontsize=8)

        for box, label, score in zip(boxes[i], labels[i], scores[i]):
            if score > threshold and label == 1:
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"Pred: {score:.2f}", color='red', fontsize=8)

    plt.tight_layout()
    plt.show()


def make_one_prediction():
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    model.eval()

    all_images = os.listdir('overtrain/train/examples')

    for name in all_images:
        image_path = os.path.join('overtrain/train/examples', name)
        image = cv2.imread(image_path)
        if image is None:
            continue 
        
        img = image.copy()
        print(f"Processing: {name}, Shape: {image.shape}")

        imTransform = transforms.ToTensor()
        image = imTransform(image)

        with torch.no_grad():
            ypred = model([image])

        bbox, label, score = ypred[0]['boxes'], ypred[0]['labels'], ypred[0]['scores']
        nums = torch.argwhere(score > 0.5).squeeze(1)
        
        if nums.numel() == 0:
            print(f"No detections for {name}")
            continue 
        
        filtered_labels = label[nums]
        categories = [weights.meta['categories'][i] for i in filtered_labels]

        for idx, i in enumerate(nums):
            x1, y1, x2, y2 = bbox[i].numpy().astype('int')
            category_name = categories[idx]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(img, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis('off')
        plt.show()


def main():
    device = get_best_device()
    model = get_model(device=device, name='vgg', path=None)
    model = train(model, 'vgg', _epochs=30, _batch_size=8, _lr=0.00001, device=device)

    dataset = CaltechPedestrianDataset(root_dir='dataset', split='test')
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    images, targets = next(iter(data_loader))
    images = [img.to(device) for img in images]

    model.eval()
    with torch.no_grad():
        predictions = model(images)

    boxes_list = [pred["boxes"].cpu().numpy() for pred in predictions]
    labels_list = [pred["labels"].cpu().numpy() for pred in predictions]
    scores_list = [pred["scores"].cpu().numpy() for pred in predictions]
    true_boxes_list = [target["boxes"].numpy() for target in targets]
    true_labels_list = [target["labels"].numpy() for target in targets]

    draw_boxes(images, boxes_list, labels_list, scores_list, true_boxes=true_boxes_list, true_labels=true_labels_list, threshold=0.2)

    map50, map =evaluate(model, data_loader, device)
    print(f"mAP50: {map50:.4f}")
    print(f"mAP@[0.5:0.95]: {map:.4f}")


if __name__ == "__main__":
    main()