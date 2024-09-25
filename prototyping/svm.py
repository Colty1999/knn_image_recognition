import torch
from torchvision import transforms, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib
from autoencoder import MushroomAutoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_autoencoder(model_path, input_channels=3, latent_dim=64):
    model = MushroomAutoencoder(input_channels=input_channels, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.eval()
    return model


def extract_features(model, data_loader):
    features = []
    labels = []
    with torch.no_grad():
        for inputs, label in data_loader:
            inputs = inputs.to(DEVICE)  # Ensure the inputs are on the correct device
            encoded = model.encode(inputs)  # Pass inputs directly without flattening
            features.append(encoded.cpu().numpy())
            labels.append(label.cpu().numpy())
    features = torch.cat([torch.tensor(f) for f in features]).numpy()
    labels = torch.cat([torch.tensor(label) for label in labels]).numpy()
    return features, labels


def visualize_latent_space(features, labels, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of the latent space")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.show()


def train_svm_classifier(features, labels, kernel='rbf'):
    svm = SVC(kernel=kernel, gamma='scale', random_state=42)
    svm.fit(features, labels)
    return svm


def evaluate_svm(svm, features, labels):
    predictions = svm.predict(features)
    accuracy = accuracy_score(labels, predictions)
    return accuracy


def save_svm_model(svm, file_path):
    joblib.dump(svm, file_path)
    print(f"SVM model saved to {file_path}")


def load_svm_model(file_path):
    svm = joblib.load(file_path)
    return svm


def main(autoencoder_path, dataset_path, svm_model_path):
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    # Load trained autoencoder
    model = load_autoencoder(autoencoder_path)

    # Extract features using the autoencoder
    features, labels = extract_features(model, data_loader)

    # Visualize the latent space
    visualize_latent_space(features, labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the SVM classifier
    svm = train_svm_classifier(X_train, y_train)

    # Evaluate the SVM classifier
    accuracy = evaluate_svm(svm, X_test, y_test)
    print(f"SVM Classifier Accuracy: {accuracy * 100:.2f}%")

    # Save the SVM model
    save_svm_model(svm, svm_model_path)


if __name__ == "__main__":
    autoencoder_path = "mushroom_autoencoder_2.pth"  # Path to the trained autoencoder model
    dataset_path = "reduced_dataset"  # Path to the dataset (same structure as before)
    svm_model_path = "svm_model.joblib"  # Path to save the trained SVM model

    main(autoencoder_path, dataset_path, svm_model_path)
