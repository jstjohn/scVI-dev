import torch
import numpy as np
from scvi.utils import no_grad, eval_modules, to_cuda
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


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

    accuracy = (torch.cat(all_y_pred) == torch.cat(all_labels)).type(torch.float32).mean().item()

    return accuracy


# The following functions require numpy arrays as inputs
def compute_accuracy_svc(data_train, data_test, labels_train, labels_test):
    # trains a SVC to predict the labels of data points in data_loader_test
    # uses grid search with plausible parameters

    # Training the classifier
    param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    svc = SVC()
    clf = GridSearchCV(svc, param_grid)
    clf.fit(data_train, labels_train)

    # Predicting the labels
    y_pred_test = clf.predict(data_test)
    y_pred_train = clf.predict(data_train)

    accuracy_train = np.mean(y_pred_train == labels_train)
    accuracy_test = np.mean(y_pred_test == labels_test)

    return accuracy_train, accuracy_test


def compute_accuracy_dt(data_train, data_test, labels_train, labels_test):
    # trains a Decision Tree to predict the labels of data points in data_loader_test
    # uses grid search with plausible parameters

    # Training the classifier
    dt = DecisionTreeClassifier()
    param_grid = {'max_depth': np.arange(3, 10)}
    clf = GridSearchCV(dt, param_grid)
    clf.fit(data_train, labels_train)

    # Predicting the labels
    y_pred_test = clf.predict(data_test)
    y_pred_train = clf.predict(data_train)

    accuracy_train = np.mean(y_pred_train == labels_train)
    accuracy_test = np.mean(y_pred_test == labels_test)

    return accuracy_train, accuracy_test


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
