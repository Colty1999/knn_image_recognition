import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from torchvision import models
from plots import visualize_tsne_with_classified_images, visualize_tsne_with_classified_images_with_prototypes
from helpers import load_data, extract_features, find_prototypical_images_paths
from prototypes import create_prototypes_mean
from knn import train_knn, train_knn_with_prototypes

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.getenv('DATA_DIR')
os.environ["LOKY_MAX_CPU_COUNT"] = os.getenv('LOKY_MAX_CPU_COUNT')

batch_size = int(os.getenv('BATCH_SIZE'))
validation_split = float(os.getenv('VALIDATION_SPLIT'))


def load_trained_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = nn.Sequential(*list(model.children())[:-1]) 
    model = model.to(device)
    model.eval()
    return model


if __name__ == '__main__':

    dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size, validation_split)
    model_path = os.getenv('MODEL_PATH')
    model = load_trained_model(model_path, num_classes=len(class_names))


    train_features, train_labels, train_image_paths  = extract_features(model, dataloaders['train'])
    val_features, val_labels, val_image_paths = extract_features(model, dataloaders['val'])

    # ["None", "Mean"]
    prototyping = "Mean"
    if (prototyping == "Mean"):
        prototypes = create_prototypes_mean(model, dataloaders['train'], class_names)

        knn = train_knn_with_prototypes(prototypes, class_names, train_features, train_labels, val_features, val_labels, n_neighbors=1)
        classified_labels = knn.predict(val_features)

        if (False):  # SAve csv with class prototypes paths
            find_prototypical_images_paths(prototypes, class_names, train_features, classified_labels, val_features, val_image_paths)

        visualize_tsne_with_classified_images_with_prototypes(prototypes, val_features, classified_labels, class_names)
    else:
        knn = train_knn(train_features, train_labels, val_features, val_labels, n_neighbors=5)
        classified_labels = knn.predict(val_features)
        visualize_tsne_with_classified_images(val_features, classified_labels, class_names)

    # plot_tsne_prototypes(prototypes, class_names)
    # plot_tsne_with_prototypes(val_features, val_labels, prototypes, class_names)
