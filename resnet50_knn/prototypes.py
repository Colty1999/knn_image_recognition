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


def load_trained_protonet(proto_model, model_path):
    """Load a pre-trained ProtoNet model."""
    proto_model.load_state_dict(torch.load(model_path))
    proto_model = proto_model.to(device)
    proto_model.eval()
    return proto_model


def create_prototypes_protonet(resnet_model, proto_model, dataloader, class_names):
    """
    Create class prototypes using the trained ProtoNet and embeddings from ResNet50.

    Args:
        resnet_model: Pre-trained ResNet50 model for feature extraction.
        proto_model: ProtoNet model to be used for generating prototypes.
        dataloader: DataLoader for the dataset to create prototypes from.
        class_names: List of class names.
        proto_model_path: Path to the pre-trained ProtoNet model.

    Returns:
        prototypes: Dictionary of class prototypes.
    """
    # Load the trained ProtoNet model
    resnet_model.eval()  # Set ResNet50 to evaluation mode
    proto_model.eval()   # Set ProtoNet to evaluation mode
    class_embeddings = defaultdict(list)  # Store embeddings per class

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for inputs, labels in dataloader:
            inputs = inputs.to(device)  # Move inputs to the device (GPU/CPU)

            # Step 1: Extract embeddings from ResNet50
            resnet_outputs = resnet_model(inputs)
            resnet_outputs = resnet_outputs.to(device)  # Ensure it's on the correct device

            # Step 2: Pass embeddings through ProtoNet to get final representations
            proto_outputs = proto_model(resnet_outputs)
            proto_outputs = proto_outputs.cpu().numpy()  # Move to CPU and convert to NumPy array

            # Accumulate embeddings for each class
            for i, label in enumerate(labels):
                class_name = class_names[label.item()]  # Get class name from label
                class_embeddings[class_name].append(proto_outputs[i])

    # Step 3: Calculate prototypes by averaging the embeddings for each class
    prototypes = {}
    for class_name, embeddings in class_embeddings.items():
        embeddings_array = np.array(embeddings)  # Convert list of embeddings to NumPy array
        prototypes[class_name] = np.mean(embeddings_array, axis=0)  # Compute mean to get the prototype for the class

    # Save prototypes to a JSON file
    with open("prototypes.json", 'w') as f:
        prototypes_to_save = {class_name: prototype.tolist() for class_name, prototype in prototypes.items()}
        json.dump(prototypes_to_save, f)

    return prototypes  # Return the prototypes for further use
