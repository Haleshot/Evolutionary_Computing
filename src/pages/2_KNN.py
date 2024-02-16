# Libraries imported
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# Defining Sigmoid function
""" 
It is especially used for models where we have to predict the probability as an output.
Since probability of anything exists only between the range of 0 and 1.

The function is differentiable that means   , we can find the slope of the sigmoid curve at any two points.
The function is monotonic but function’s derivative is not.
"""
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Defining a function for forward pass in neural network
"""
This function performs a forward pass in neural network and gives output as neural network with forward pass.
By using parameters : 
-> nn_model: Neural Network model
-> data: Input Dataset
"""
def forward_pass(nn_model, data):
    nn_model.forward(data[:, 1:])
    return nn_model.output

# Defining a function train_classifier_knn
"""
This function trains a k-Nearest Neighbors (kNN) classifier using parameters:
-> train_points: data point of training set
-> train_labels: Labels corresponding to each data point
-> k_neighbors: Number of neighbors for kNN

As output it gives train knn classifier
"""
def train_classifier_knn(train_points, train_labels, k_neighbors=10):
    knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn_classifier.fit(train_points, train_labels)
    return knn_classifier

# Defining a function data_visualization
"""
This function helps in visualization of data through a scatter plot.
By using parameters :
-> data_points: points from dataset
-> data_labels: Labels corresponding to each data point
-> N: Number of unique labels

Output is in a form of scatter plot.
"""
def data_visualization(data_points, data_labels, N=4):
    # Create a 1x1 subplot with a specified figure size
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # Define a color map using the 'jet' colormap
    cmap = plt.cm.jet
    # Create a list of colors based on the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # Create a new colormap from the list of colors
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # Define bins and normalize for color mapping
    bounds = np.linspace(0, N, N + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # Extract x and y coordinates
    x, y = data_points[:, 0], data_points[:, 1]
    # Generate scatter plot with color codes
    scatter = ax.scatter(x, y, c=data_labels, cmap=cmap, norm=norm)
    # Add a colorbar to the plot with specified boundaries and ticks
    plt.colorbar(scatter, ax=ax, boundaries=bounds, ticks=np.arange(N))
    plt.show()

# Defining a function evaluate_nn
"""
This function helps to evaluate the neural network model and 
it gives output as Accuracy and F1 score.
It assess the performance of a neural network by combining its output with 
k-Nearest Neighbors (kNN) classification on a test set and returning accuracy and F1 score.

By using parameters:
-> nn_model: Neural Network model
-> test_data: Test dataset
-> train_data: Training dataset
-> k_neighbors: Number of neighbors for kNN
"""
def evaluate_nn(nn_model, test_data, train_data, k_neighbors=10):
    # Perform a forward pass on the training data using the neural network
    train_points = forward_pass(nn_model, train_data)
    # Perform a forward pass on the test data using the neural network
    test_points = forward_pass(nn_model, test_data)
    # Train a knn classifier
    knn_classifier = train_classifier_knn(train_points, train_data[:, 0], k_neighbors)
    # Predict labels
    y_pred = knn_classifier.predict(test_points)
    # Calculate Accuracy
    accuracy = metrics.accuracy_score(test_data[:, 0], y_pred)
    # Calculate F1 score
    f1_score = metrics.f1_score(test_data[:, 0], y_pred, average='macro', labels=np.unique(y_pred))

    return accuracy, f1_score