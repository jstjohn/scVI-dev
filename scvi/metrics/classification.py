from collections import namedtuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from scvi.utils import no_grad, eval_modules, to_cuda

Accuracy = namedtuple('Accuracy', ['weighted', 'unweighted', 'worst'])


def full_accuracy(accuracy_classes, classes_probabilities):
    return Accuracy(weighted=compute_weighted_accuracy(accuracy_classes),
                    unweighted=compute_unweighted_accuracy(accuracy_classes, classes_probabilities),
                    worst=compute_worst_accuracy(accuracy_classes)
                    )


def compute_accuracy_classes(y, labels):
    n_labels = len(np.unique(labels))
    accuracy_classes = np.array([np.mean((labels[labels == cl] == y[labels == cl])) for cl in range(n_labels)])
    classes_probabilities = np.array([np.mean(labels == cl) for cl in range(n_labels)])
    return accuracy_classes, classes_probabilities


def compute_unweighted_accuracy(accuracy_classes, classes_probabilities):
    return np.dot(accuracy_classes, classes_probabilities)


def compute_weighted_accuracy(accuracy_classes):
    return np.mean(accuracy_classes)


def compute_worst_accuracy(accuracy_classes):
    return np.min(accuracy_classes)


@no_grad()
@eval_modules()
def compute_accuracy(vae, data_loader, classifier=None):
    all_y_pred = []
    all_labels = []
    for i_batch, tensors in enumerate(data_loader):
        if vae.use_cuda:
            tensors = to_cuda(tensors)
        sample_batch, _, _, _, labels = tensors
        sample_batch = sample_batch.type(torch.float32)
        all_labels += [labels.view(-1)]

        if classifier is not None:
            # Then we use the specified classifier
            mu_z, _, _ = vae.z_encoder(sample_batch)
            y_pred = classifier(mu_z).argmax(dim=-1)
        else:
            # Then the vae must implement a classify function
            y_pred = vae.classify(sample_batch).argmax(dim=-1)
        all_y_pred += [y_pred]

    all_y_pred = torch.cat(all_y_pred).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    accuracy_classes, classes_probabilities = compute_accuracy_classes(all_y_pred, all_labels)
    return full_accuracy(accuracy_classes, classes_probabilities)


# The following functions require numpy arrays as inputs
def compute_accuracy_svc(data_train, data_test, labels_train, labels_test, unit_test=False):
    # trains a SVC to predict the labels of data points in data_loader_test
    # uses grid search with plausible parameters

    # Training the classifier
    # More hyper-parameters than for the scMAP paper (they only use linear kernel also
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
    if unit_test:
        param_grid = [{'C': [1], 'kernel': ['linear']}]
    svc = SVC()
    clf = GridSearchCV(svc, param_grid)
    clf.fit(data_train, labels_train)

    # Predicting the labels
    y_pred_test = clf.predict(data_test)
    y_pred_train = clf.predict(data_train)

    return full_accuracy(*compute_accuracy_classes(y_pred_train, labels_train)), \
           full_accuracy(*compute_accuracy_classes(y_pred_test, labels_test))


def compute_accuracy_rf(data_train, data_test, labels_train, labels_test, unit_test=False):
    # trains a Decision Tree to predict the labels of data points in data_loader_test
    # uses grid search with plausible parameters

    # Training the classifier
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    # Definitely more hyper-parameters than in scMAP (they only have n_estimators = 50
    # and the rest set to default
    param_grid = {'max_depth': np.arange(3, 10), 'n_estimators': [10, 50, 100, 200]}
    if unit_test:
        param_grid = [{'max_depth': [3], 'n_estimators': [10]}]
    clf = GridSearchCV(rf, param_grid)
    clf.fit(data_train, labels_train)

    # Predicting the labels
    y_pred_test = clf.predict(data_test)
    y_pred_train = clf.predict(data_train)

    return full_accuracy(*compute_accuracy_classes(y_pred_train, labels_train)), \
           full_accuracy(*compute_accuracy_classes(y_pred_test, labels_test))


def compute_accuracy_md(data_train_latent, data_test_latent, labels_train, labels_test, n_labels):
    # uses clustering and Majority Decision to predict the labels of data points in data_loader_test

    split_index = len(data_train_latent)
    X = np.concatenate((data_train_latent, data_test_latent))
    # Cluster the data using k-means
    kmeans = KMeans(n_clusters=n_labels).fit(X)
    clusters_train = kmeans.predict(X[:split_index])
    clusters_test = kmeans.predict(X[split_index:])

    # Use Majority decision to attribute a label to each cluster
    clusters_labels_assignment = np.zeros((n_labels, n_labels))
    for idx in range(len(clusters_train)):
        clusters_labels_assignment[clusters_train[idx], labels_train[idx]] += 1
    clusters_labels = np.argmax(clusters_labels_assignment, axis=1)

    # Compute accuracy for train
    accuracy_train = 0
    for idx in range(len(clusters_train)):
        # If the majoritary label in the sample's cluster is the true label,
        # then the prediction is good
        if clusters_labels[clusters_train[idx]] == labels_train[idx]:
            accuracy_train += 1
    # Compute accuracy for test
    accuracy_test = 0
    for idx in range(len(clusters_test)):
        if clusters_labels[clusters_test[idx]] == labels_test[idx]:
            accuracy_test += 1
    accuracy_train /= len(labels_train)
    accuracy_test /= len(labels_test)
    return accuracy_train, accuracy_test
