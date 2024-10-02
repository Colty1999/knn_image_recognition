import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import pandas as pd
from plots import plot_accuracy
from helpers import load_data
from prototypical_loss import prototypical_loss  # Import your Prototypical loss function

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.getenv('DATA_DIR')
# os.environ["LOKY_MAX_CPU_COUNT"] = os.getenv('LOKY_MAX_CPU_COUNT')

num_epochs = int(os.getenv('NUM_EPOCHS'))
learning_rate = float(os.getenv('LEARNING_RATE'))
validation_split = float(os.getenv('VALIDATION_SPLIT'))
# n_support = int(os.getenv('N_SUPPORT'))  # Number of support examples per class
n_support = 5  # Number of support examples per class


def initialize_model():
    # Load ResNet50 without final FC layer to use as a feature extractor
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    # model.fc = nn.Identity()  # Remove the classification head to use the embeddings
    # model.fc = nn.Linear(num_ftrs, num_classes)
    model.fc = nn.Identity() 
    model.final_conv_layer = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        # nn.Conv2d(512, 256, kernel_size=3, padding=1),
        # nn.BatchNorm2d(256),
        # nn.ReLU()
    )
    return model


def train_model_protonet(model, dataloaders, dataset_sizes, optimizer, n_support, num_epochs=25):
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0.0

            with tqdm(total=len(dataloaders[phase]), desc=f'{phase} epoch {epoch}') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # Forward pass to get feature embeddings from ResNet50
                        embeddings = model(inputs)

                        # Compute Prototypical Loss (dynamically computes prototypes)
                        loss, acc = prototypical_loss(embeddings, labels, n_support)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_acc += acc.item()

                    pbar.update(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / len(dataloaders[phase])

            if phase == 'train':
                epoch_train_loss = epoch_loss
                epoch_train_acc = epoch_acc
            else:
                epoch_val_loss = epoch_loss
                epoch_val_acc = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # Now we update the history dictionary after both phases are done
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print()

    # Convert history into DataFrame for easier handling
    history_df = pd.DataFrame(history)
    print(history_df)  # Print the table of accuracies and losses

    return model, history_df


if __name__ == '__main__':
    dataloaders, dataset_sizes, class_names = load_data(data_dir, validation_split)
    model = initialize_model()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs for training")

        total_memory = 0
        for i in range(num_gpus):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            total_memory += gpu_memory

        total_memory_gb = total_memory / (1024 ** 3)
        print(f"Total available GPU memory: {total_memory_gb:.2f} GB")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train the ResNet50 model with Prototypical Loss
    model, history_df = train_model_protonet(model, dataloaders, dataset_sizes, optimizer, n_support, num_epochs=num_epochs)

    # Plot accuracy over epochs
    plot_accuracy(history_df)

    torch.save(model.state_dict(), os.getenv('PROTONET_MODEL_PATH'))
