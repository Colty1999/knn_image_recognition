import os
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from plots import visualize_tsne_with_classified_images, visualize_tsne_with_classified_images_with_prototypes
from helpers import load_data, extract_features
from prototypes import create_prototypes_mean, create_prototypes_protonet
from knn import train_knn, train_knn_with_prototypes, train_knn_with_prototypes_protonet

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.getenv('DATA_DIR')
os.environ["LOKY_MAX_CPU_COUNT"] = os.getenv('LOKY_MAX_CPU_COUNT')

batch_size = int(os.getenv('BATCH_SIZE'))
validation_split = float(os.getenv('VALIDATION_SPLIT'))


def load_trained_model_classic(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model


def prepare_classic():
    dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size, validation_split)
    model_path = "Resnet_model.pth"
    model = load_trained_model_classic(model_path, num_classes=len(class_names))

    train_features, train_labels = extract_features(model, dataloaders['train'])
    val_features, val_labels = extract_features(model, dataloaders['val'])

    return dataloaders, dataset_sizes, class_names, model, train_features, train_labels, val_features, val_labels


def load_trained_model_protonet():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model.final_conv_layer = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
    )
    # model_path = os.getenv('PROTONET_MODEL_PATH')
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict)

    model_path = os.getenv('PROTONET_MODEL_PATH')
    state_dict = torch.load(model_path)

    # Remove the 'module.' prefix if it was trained with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    # Load the modified state dictionary into the model
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    return model


def extract_features_with_protonet(model, dataloader):
    """Extracts features using the Prototypical Network model."""
    device = next(model.parameters()).device  # Ensure model and inputs are on the same device
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            outputs = model(inputs)  # Extract features (ensure this matches ResNet50's feature dimensions)
            features.append(outputs.cpu().numpy())  # Convert to numpy array for KNN
            labels.append(targets.cpu().numpy())  # Convert labels to numpy array

    features = np.vstack(features)  # Stack all features into one numpy array
    labels = np.hstack(labels)  # Stack all labels into one numpy array

    return features, labels

def create_prototypes_protonet(model, dataloader, class_names):
    """Create prototypes by averaging the features of the support samples for each class."""
    # Get the model's device from the model parameters
    device = next(model.parameters()).device
    class_prototypes = {}

    # Extract features for each class
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)  # Move inputs to the same device as the model
            outputs = model(inputs)
            for feature, target in zip(outputs.cpu(), targets.cpu()):
                class_name = class_names[target.item()]
                if class_name not in class_prototypes:
                    class_prototypes[class_name] = []
                class_prototypes[class_name].append(feature.numpy())
    
    # Average the features for each class to create the prototype
    for class_name, features in class_prototypes.items():
        class_prototypes[class_name] = np.mean(features, axis=0)

    return class_prototypes


def prepare_protonet():
    # Load the data using the same dataloader as in prepare_classic
    dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size, validation_split)
    
    # Load the trained Prototypical Network model
    model = load_trained_model_protonet()  # This loads your trained Protonet model

    # Extract features for both train and validation sets using the Prototypical Network
    train_features, train_labels = extract_features_with_protonet(model, dataloaders['train'])
    val_features, val_labels = extract_features_with_protonet(model, dataloaders['val'])

    return dataloaders, dataset_sizes, class_names, model, train_features, train_labels, val_features, val_labels



if __name__ == '__main__':    
    # ["None", "Mean", "Protonet"]
    prototyping = "Protonet"
    if (prototyping == "Mean"):
        dataloaders, dataset_sizes, class_names, model, train_features, train_labels, val_features, val_labels = prepare_classic()
        prototypes = create_prototypes_mean(model, dataloaders['train'], class_names)

        knn = train_knn_with_prototypes(prototypes, class_names, train_features, train_labels, val_features, val_labels, n_neighbors=1)
        classified_labels = knn.predict(val_features)
        visualize_tsne_with_classified_images_with_prototypes(prototypes, val_features, classified_labels, class_names)
    elif (prototyping == "Protonet"):
        model = load_trained_model_protonet()
        
        # Prepare data using the existing function `prepare_classic` (assuming it prepares dataloaders)
        dataloaders, dataset_sizes, class_names, _, train_features, train_labels, val_features, val_labels = prepare_protonet()

        # Create prototypes from the training data
        prototypes = create_prototypes_protonet(model, dataloaders['train'], class_names)
        
        # Train KNN using prototypes
        knn = train_knn_with_prototypes_protonet(prototypes, class_names, train_features, train_labels, val_features, val_labels, n_neighbors=1)
        
        # Classify validation data
        classified_labels = knn.predict(val_features)

        # Visualize the results (assuming you have a similar function as before)
        visualize_tsne_with_classified_images_with_prototypes(prototypes, val_features, classified_labels, class_names)
    else:
        dataloaders, dataset_sizes, class_names, model, train_features, train_labels, val_features, val_labels = prepare_classic()
        knn = train_knn(train_features, train_labels, val_features, val_labels, n_neighbors=5)
        classified_labels = knn.predict(val_features)
        visualize_tsne_with_classified_images(val_features, classified_labels, class_names)

    # plot_tsne_prototypes(prototypes, class_names)
    # plot_tsne_with_prototypes(val_features, val_labels, prototypes, class_names)
