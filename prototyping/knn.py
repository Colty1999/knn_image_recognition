import torch
from torchvision import transforms, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from autoencoder import MushroomAutoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_autoencoder(model_path, input_channels=3, latent_dim=128):
    model = MushroomAutoencoder(input_channels=input_channels, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.eval()
    return model


def extract_features(model, data_loader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in data_loader:
            inputs = inputs.to(DEVICE)
            encoded = model.encode(inputs)
            features.append(encoded.cpu().numpy())
            labels.append(label.cpu().numpy())
    features = torch.cat([torch.tensor(f) for f in features]).numpy()
    labels = torch.cat([torch.tensor(label) for label in labels]).numpy()
    return features, labels


def train_knn_classifier(features, labels, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)
    return knn


def evaluate_knn(knn, features, labels):
    predictions = knn.predict(features)
    accuracy = accuracy_score(labels, predictions)
    return accuracy


def save_knn_model(knn, file_path):
    joblib.dump(knn, file_path)
    print(f"KNN model saved to {file_path}")


def load_knn_model(file_path):
    knn = joblib.load(file_path)
    return knn


def main(autoencoder_path, dataset_path, knn_model_path, k=5):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    model = load_autoencoder(autoencoder_path)

    features, labels = extract_features(model, data_loader)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    knn = train_knn_classifier(X_train, y_train, k)

    accuracy = evaluate_knn(knn, X_test, y_test)
    print(f"KNN Classifier Accuracy: {accuracy * 100:.2f}%")

    save_knn_model(knn, knn_model_path)


if __name__ == "__main__":
    autoencoder_path = "mushroom_autoencoder_2.pth"
    dataset_path = "reduced_dataset"
    knn_model_path = "knn_model.joblib"

    main(autoencoder_path, dataset_path, knn_model_path)
