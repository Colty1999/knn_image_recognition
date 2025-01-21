import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from helpers import train_transform, prototype_transform

class MushroomDataset(Dataset):
    def __init__(self, root_dir, csv_file, prototype_sampling_rate=0.005):
        self.root_dir = root_dir
        self.prototype_df = pd.read_csv(csv_file)
        self.proto_rate = prototype_sampling_rate

        # Collect image paths and class names
        self.image_paths = []
        self.class_names = []
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                for img in os.listdir(class_folder):
                    self.image_paths.append(os.path.join(class_folder, img))
                    self.class_names.append(class_name)

        # Define a transform to convert PIL images to tensors
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = self.class_names[idx]
        
        # Load the image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Retrieve and load the prototype image for this class
        prototype_path = self.prototype_df[self.prototype_df['Class'] == class_name]['Closest Image Path'].values[0]
        prototype_image = Image.open(prototype_path).convert("RGB")

        # Apply transformations to the single image and prototype
        image = train_transform(image)
        prototype_image = prototype_transform(prototype_image)

        # Convert images to tensors
        image = self.to_tensor(image)
        prototype_image = self.to_tensor(prototype_image)

        # print(f"Image: {img_path}, Prototype: {prototype_path}, Class: {class_name}")
        return image, prototype_image, class_name
