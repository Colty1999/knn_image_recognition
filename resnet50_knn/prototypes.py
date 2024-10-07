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
    pass