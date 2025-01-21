import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import pandas as pd
from plots import plot_accuracy, plot_loss
from helpers import load_data

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.getenv('DATA_DIR')
os.environ["LOKY_MAX_CPU_COUNT"] = os.getenv('LOKY_MAX_CPU_COUNT')

batch_size = int(os.getenv('BATCH_SIZE'))
num_epochs = int(os.getenv('NUM_EPOCHS'))
learning_rate = float(os.getenv('LEARNING_RATE'))
validation_split = float(os.getenv('VALIDATION_SPLIT'))


def initialize_model(num_classes):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_train_loss = 0.0
        epoch_train_corrects = 0
        epoch_val_loss = 0.0
        epoch_val_corrects = 0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            with tqdm(total=len(dataloaders[phase]), desc=f'{phase} epoch {epoch}') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    pbar.update(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                epoch_train_loss = epoch_loss
                epoch_train_corrects = epoch_acc
            else:
                epoch_val_loss = epoch_loss
                epoch_val_corrects = epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        # Now we update the history dictionary after both phases are done
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_corrects.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_corrects.item())

    # Convert history into DataFrame for easier handling
    history_df = pd.DataFrame(history)
    print(history_df)  # Print the table of accuracies and losses

    return model, history_df


if __name__ == '__main__':
    dataloaders, dataset_sizes, class_names = load_data(data_dir, batch_size, validation_split)
    model = initialize_model(num_classes=len(class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Step 1: Train the ResNet50 model
    model, history_df = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=num_epochs)

    # Step 5: Plot accuracy over epochs
    plot_accuracy(history_df)

    # Plot loss over epochs
    plot_loss(history_df)

    # Save the history DataFrame to a CSV file
    os.makedirs('plots', exist_ok=True)  # Ensure the plots directory exists
    history_csv_path = os.path.join('plots', 'training_history.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")

    torch.save(model.state_dict(), os.getenv('MODEL_PATH'))
