import torch
import numpy as np
from collections import defaultdict
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_prototypes_mean(model, dataloader, class_names):
    model.eval()
    class_embeddings = defaultdict(list)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()

            for i, label in enumerate(labels):
                class_embeddings[class_names[label.item()]].append(outputs[i])

    prototypes = {}
    for class_name, embeddings in class_embeddings.items():
        prototypes[class_name] = np.mean(embeddings, axis=0)

    with open("prototypes.json", 'w') as f:
        prototypes_to_save = {class_name: prototype.tolist() for class_name, prototype in prototypes.items()}
        json.dump(prototypes_to_save, f)

    return prototypes


def find_closest_image_to_prototype(prototype, features, image_paths):
    """
    Find the image that is closest to the given prototype.

    :param prototype: The mean prototype of a class.
    :param features: The extracted features of all images.
    :param image_paths: List of paths of the images in the dataset.
    :return: Path of the image that is closest to the prototype.
    """
    distances = np.linalg.norm(features - prototype, axis=1)
    closest_index = np.argmin(distances)
    return image_paths[closest_index]