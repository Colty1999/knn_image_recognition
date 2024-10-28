import os
import pandas as pd
import random
from torch.utils.data import Dataset
from PIL import Image
from helpers import train_transform, prototype_transform

class MushroomDataset(Dataset):
    def __init__(self, root_dir, csv_file, prototype_sampling_rate=0.005):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the CSV file with class-prototype pairs.
            prototype_sampling_rate (float): The rate of prototype sampling to replace the input image with the prototype.
        """
        self.root_dir = root_dir
        self.prototype_df = pd.read_csv(csv_file)
        self.proto_rate = prototype_sampling_rate

        # Create a list of all image paths in the dataset
        self.image_paths = []
        self.class_names = []
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if os.path.isdir(class_folder):
                for img in os.listdir(class_folder):
                    self.image_paths.append(os.path.join(class_folder, img))
                    self.class_names.append(class_name)
        # print("Class names len: ", self.class_names)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path and class name
        img_path = self.image_paths[idx]
        class_name = self.class_names[idx]
        
        # Load the image
        image = Image.open(img_path).convert("RGB")
        
        # Get the corresponding prototype image from the CSV file
        prototype_path = self.prototype_df[self.prototype_df['Class'] == class_name]['Closest Image Path'].values[0]
        prototype_image = Image.open(prototype_path).convert("RGB")

        # Apply training transforms to the image
        image = train_transform(image)

        # Apply prototype transforms to the prototype image
        prototype_image = prototype_transform(prototype_image)

        return image, prototype_image, class_name
