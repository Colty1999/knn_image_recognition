from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


# Function to visualize t-SNE for prototypes only
def plot_tsne_prototypes(prototypes, class_names, perplexity=30, n_iter=1000):
    print("Running t-SNE on the prototypes...")

    # Convert prototypes to a numpy array
    prototype_features = np.array(list(prototypes.values()))

    # Apply t-SNE on prototypes
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    prototypes_2d = tsne.fit_transform(prototype_features)

    # Use a colormap with more colors for all classes
    cmap = plt.get_cmap("gist_ncar")

    # Plot t-SNE projection of prototypes
    plt.figure(figsize=(10, 6))

    # Plot prototypes with star markers
    for i, (class_name, prototype) in enumerate(zip(prototypes.keys(), prototypes_2d)):
        plt.scatter(prototype[0], prototype[1], color=cmap(i / len(class_names)), marker='*', s=200, edgecolor='black', label=f'Prototype {class_name}')

    plt.title('t-SNE projection of prototypes only')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True)
    plt.legend(loc="best", bbox_to_anchor=(1, 1))
    plt.show()


def plot_tsne_with_prototypes(features, labels, prototypes, class_names, perplexity=30, n_iter=1000):
    print("Running t-SNE on the feature space...")

    # Combine data points and prototypes for t-SNE transformation
    prototype_features = np.array(list(prototypes.values()))
    combined_features = np.vstack([features, prototype_features])

    # Apply t-SNE to both the data points and prototypes at the same time
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    combined_2d = tsne.fit_transform(combined_features)

    # Separate the t-SNE-transformed features for data points and prototypes
    features_2d = combined_2d[:len(features)]
    prototypes_2d = combined_2d[len(features):]

    # Use a colormap with more colors for all classes
    cmap = plt.get_cmap("gist_ncar")

    # Plot t-SNE projection of data points
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, s=10)

    # Plot prototypes with star markers
    for i, (class_name, prototype) in enumerate(zip(prototypes.keys(), prototypes_2d)):
        plt.scatter(prototype[0], prototype[1], color=cmap(i / len(class_names)), marker='*', s=200, edgecolor='black', label=f'Prototype {class_name}')

    plt.title('t-SNE projection of the feature space with aligned prototypes')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True)
    plt.legend(loc="best", bbox_to_anchor=(1, 1))
    plt.show()


# Plot accuracy over epochs
def plot_accuracy(history_df):
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_tsne_with_classified_images(new_features, new_labels, class_names):
    # Perform t-SNE transformation only on the new features
    tsne = TSNE(n_components=2, random_state=42)
    new_features_2d = tsne.fit_transform(new_features)

    # Create a color map that assigns different colors to each class
    unique_classes = np.unique(new_labels)
    colors = plt.cm.get_cmap('jet', len(unique_classes))

    # Plot new classified images
    for i, class_label in enumerate(unique_classes):
        class_mask = new_labels == class_label
        plt.scatter(new_features_2d[class_mask, 0], new_features_2d[class_mask, 1],
                    color=colors(i), label=f"Class {class_names[class_label]}", alpha=0.6)

    # Add colorbar
    plt.colorbar()

    # Move the legend to the right after the colorbar
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title("t-SNE of Classified Images")
    plt.show()


def visualize_tsne_with_classified_images_with_prototypes(prototypes, new_features, new_labels, class_names):
    # If prototypes is a dict, convert it to an array in the correct order of class names
    if isinstance(prototypes, dict):
        prototypes = np.array([prototypes[class_name] for class_name in class_names])

    # Ensure prototypes are 2D and match the dimension of new_features
    if len(prototypes.shape) == 1:
        prototypes = np.expand_dims(prototypes, axis=0)  # Add a dimension if necessary

    # Check if both prototypes and new features have the same number of dimensions
    if prototypes.shape[1] != new_features.shape[1]:
        raise ValueError(f"Prototypes and new features must have the same number of dimensions, "
                         f"but got {prototypes.shape[1]} and {new_features.shape[1]} respectively.")

    # Combine prototypes and new features
    all_features = np.vstack([prototypes, new_features])  # Combine prototypes and new features

    # Perform t-SNE transformation
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)

    num_prototypes = len(prototypes)

    # Separate 2D points for plotting
    prototypes_2d = all_features_2d[:num_prototypes]
    new_features_2d = all_features_2d[num_prototypes:]

    # Create a color map that assigns the same color to the prototypes and their respective images
    unique_classes = np.unique(new_labels)
    colors = plt.cm.get_cmap('jet', len(unique_classes))

    # Plot new classified images
    for i, class_label in enumerate(unique_classes):
        class_mask = new_labels == class_label

        # Convert class_label to integer index if necessary
        if isinstance(class_label, np.str_):  # If class_label is a string
            class_label_idx = class_names.index(class_label)  # Get the index of the class label
        else:
            class_label_idx = class_label  # It's already an integer

        plt.scatter(new_features_2d[class_mask, 0], new_features_2d[class_mask, 1],
                    color=colors(i), label=f"Class {class_names[class_label_idx]}", alpha=0.6)

    # Plot prototypes with the same color as their class
    for i, prototype in enumerate(prototypes_2d):
        plt.scatter(prototype[0], prototype[1], color=colors(i), marker='*', s=200, edgecolor='black',
                    label=f'Prototype {class_names[i]}')

    # Add colorbar
    plt.colorbar()

    # Move the legend to the right after the colorbar
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title("t-SNE of Prototypes and Classified Images")
    plt.show()
