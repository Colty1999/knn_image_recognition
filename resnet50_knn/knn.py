from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time
# from collections import Counter


def train_knn(train_features, train_labels, val_features, val_labels, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    start_time_fit = time.time()
    knn.fit(train_features, train_labels)
    end_time_fit = time.time()

    start_time_score = time.time()
    accuracy = knn.score(val_features, val_labels)
    end_time_score = time.time()

    print(f"KNN Classifier Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken to fit the model: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to score the model: {end_time_score - start_time_score:.4f} seconds")

    return knn


def train_knn_with_prototypes(prototypes, class_names, train_features, train_labels, val_features, val_labels, n_neighbors=1):
    unique_prototypes = np.array([prototypes[class_name] for class_name in class_names])
    unique_labels = np.array([class_name for class_name in class_names])
    val_labels = np.array([class_names[label] for label in val_labels])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    start_time_fit = time.time()
    knn.fit(unique_prototypes, unique_labels)
    end_time_fit = time.time()

    start_time_score = time.time()
    accuracy = knn.score(val_features, val_labels)
    end_time_score = time.time()

    # predicted_labels = knn.predict(val_features)
    # label_counts = Counter(predicted_labels)
    # print("\nClassified images per class:")
    # for class_label, count in label_counts.items():
    #     print(f"Class {class_label}: {count} images")

    print(f"KNN Classifier with Prototypes Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken to fit the model: {end_time_fit - start_time_fit:.4f} seconds")
    print(f"Time taken to score the model: {end_time_score - start_time_score:.4f} seconds")

    return knn
