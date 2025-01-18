from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from PIL import Image
import torch
from torchvision.io.image import decode_image


class CaltechPedestrianDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split

        self.transform = transforms.Compose([
            # transforms.Resize((320, 320)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

        self.examples_dir = os.path.join(self.root_dir, self.split, "examples")
        self.annotations_dir = os.path.join(self.root_dir, self.split, "annotations")

        self.images = sorted(os.listdir(self.examples_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.examples_dir, image_name)
        annotation_path = os.path.join(self.annotations_dir, image_name.replace(".png", ".txt"))

        image = Image.open(image_path).convert('RGB')
        # image = decode_image(image_path)
        width, height = image.size 

        boxes = []
        labels = []

        image = self.transform(image)

        with open(annotation_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                xmin = int((x_center - box_width / 2) * width)
                ymin = int((y_center - box_height / 2) * height)
                xmax = int((x_center + box_width / 2) * width)
                ymax = int((y_center + box_height / 2) * height)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        

        return image, {'boxes': boxes, 'labels': labels}