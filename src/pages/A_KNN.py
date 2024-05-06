import streamlit as st

# st.set_page_config(
#     page_title="K-Nearest Neighbor (KNN) Implementation 🧑‍💻",
#     layout="wide"
# ) # TODO : Define one in the Introduction file.

st.markdown(

    """
    ## Overview
    
    This Python script includes functions related to a neural network model evaluation using k-Nearest Neighbors (kNN) and data visualization. The functions cover the definition of a sigmoid function, forward pass in a neural network, training a kNN classifier, visualizing data, and evaluating a neural network's performance.
    
    ## Contents
    
    1. **Sigmoid Function Definition:**
       - Defines a sigmoid function for models predicting probabilities between 0 and 1. The function is explained for its use and characteristics.
    
    2. **Forward Pass in Neural Network:**
       - Defines a function for the forward pass in a neural network, producing the output after the pass.
    
    3. **Train kNN Classifier Function:**
       - Defines a function to train a kNN classifier using specified parameters.
    
    4. **Data Visualization Function:**
       - Defines a function for visualizing data through a scatter plot, with parameters for data points, labels, and unique labels.
    
    5. **Evaluate Neural Network Function:**
       - Evaluates a neural network's performance using kNN classification on a test set. Returns accuracy and F1 score.
       
    ## Usage
    To utilize the k-Nearest Neighbor (kNN) implementation for classification tasks, follow these steps:

    1. Import the necessary libraries and classes from the provided Python file into your project.
    ```python
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    import matplotlib.pyplot as plt
    ```

    2. Prepare dataset for classification, ensuring properl formatted with features and corresponding labels.

    3. Train kNN classifier by initializing an instance of the KNeighborsClassifier class and specifying parameters.

    4. Fit the classifier to training data using the `fit` method.

    5. Predict labels for new data points using the predict method.

    6. Evaluate the performance of classifier using metrics such as accuracy and F1 score.
    
    The `sigmoid` function applies the sigmoid function to the given input array or scalar.
    
    - **Inputs:**
        - X: Input data or values.
    
    - **Functionality:**
        - Applies the sigmoid activation function element-wise to the input array or scalar.
    
    ```python
    def sigmoid(Z):
        '''
        Calculates the sigmoid function for the given input.

        Args:
            Z (numpy.ndarray): Input to the sigmoid function.
        
        Returns:
            numpy.ndarray: Output of the sigmoid function.
        '''    
        return 1 / (1 + np.exp(-Z))
    ```
    
    ---
    The function `compute_accuracy_with_knn_partitioning` evaluates the performance of a trained neural network classifier by integrating K-nearest neighbors (KNN) partitioning, providing insights into its accuracy and F1 score.
    - **Inputs:**
        - `neural_network` (NeuralNetwork): The trained neural network model.
        - `test_data` (numpy.ndarray): Test data with labels.
        - `train_data` (numpy.ndarray): Training data with labels.

    - **Functionality:**
        - Perform forward pass for both training and test data through the neural network to obtain the output points.
        - Initialize a KNN classifier with 10 neighbors and fit it to the training points obtained from the neural network.
        - Use the trained KNN classifier to predict labels for the test points.
        - Compute the accuracy score by comparing the predicted labels with the actual labels of the test data.
        - Compute the F1 score, which is the harmonic mean of precision and recall, using the predicted labels and actual labels.
        - Visualize the training data points in a scatter plot, where each point is colored according to its label.
    
    - **Return:**
        - Type: tuple
        - Description: A tuple containing the accuracy score and the F1 score calculated for the neural network classifier using KNN partitioning.

    ```python
    def compute_accuracy_with_knn_partitioning(neural_network, test_data, train_data):
        '''
        Compute the accuracy of a neural network classifier using KNN partitioning.
    
        Args:
            neural_network (NeuralNetwork): The trained neural network.
            test_data (numpy.ndarray): Test data with labels.
            train_data (numpy.ndarray): Training data with labels.
    
        Returns:
            tuple: A tuple containing accuracy and F1 score.
        '''
    
        # Perform forward pass for training data
        neural_network.forward(train_data[:, 1:])  
        # Get output from neural network
        train_points = neural_network.output  
    
        # Perform forward pass for test data
        # Exclude labels from input
        neural_network.forward(test_data[:, 1:])  
        # Get output from neural network
        test_points = neural_network.output  
    
        # Fit KNN classifier on training data
        # Create KNN classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=10)  
        # Train KNN classifier on training data
        knn_classifier.fit(train_points, train_data[:, 0])  
    
        # Predict labels for test data using trained KNN classifier
        predicted_labels = knn_classifier.predict(test_points)  
    
        # Calculate accuracy and F1 score
        accuracy = metrics.accuracy_score(test_data[:, 0], predicted_labels)  
        # Calculate F1 score
        f1_score = metrics.f1_score(test_data[:, 0], predicted_labels, average='macro', labels=np.unique(predicted_labels))  
    
        # Visualize the data with a scatter plot
        # Get the number of unique classes
        num_classes = len(np.unique(train_data[:, 0]))  
        # Create figure and axis for plotting
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))  
        # x-coordinates for scatter plot
        x = train_points[:, 0] 
        # y-coordinates for scatter plot 
        y = train_points[:, 1]  
        # Labels for coloring points
        labels = train_data[:, 0]  
        
        # Choose colormap
        colormap = plt.cm.jet
        # Create list of colors from colormap 
        cmap_list = [colormap(i) for i in range(colormap.N)] 
        # Create custom colormap 
        colormap = colormap.from_list('Custom cmap', cmap_list, colormap.N)  
        
        # Define boundaries for colormap
        bounds = np.linspace(0, num_classes, num_classes + 1)  
        # Normalize colormap
        norm = plt.Normalize(0, num_classes)  
        
        # Create scatter plot
        scatter = ax.scatter(x, y, c=labels, cmap=colormap, norm=norm) 
        # Create colorbar 
        colorbar = plt.colorbar(scatter, spacing='proportional', ticks=bounds) 
        # Set label for colorbar 
        colorbar.set_label('Custom colorbar')  
        plt.show()  
    
        # Return accuracy and F1 score
        return accuracy, f1_score
        """)









############################################## CODE IMPLEMENTATION ##############################################################


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt



def sigmoid(Z):
    '''
    Calculates the sigmoid function for the given input.

    Args:
        Z (numpy.ndarray): Input to the sigmoid function.

    Returns:
        numpy.ndarray: Output of the sigmoid function.
    '''    
    return 1 / (1 + np.exp(-Z))


# def compute_accuracy_with_knn_partitioning(neural_network, test_data, train_data):
#     '''
#     Compute the accuracy of a neural network classifier using KNN partitioning.

#     Args:
#         neural_network (NeuralNetwork): The trained neural network.
#         test_data (numpy.ndarray): Test data with labels.
#         train_data (numpy.ndarray): Training data with labels.

#     Returns:
#         tuple: A tuple containing accuracy and F1 score.
#     '''

#     # Perform forward pass for training data
#     neural_network.forward(train_data[:, 1:])  
#     # Get output from neural network
#     train_points = neural_network.output  

#     # Perform forward pass for test data
#     # Exclude labels from input
#     neural_network.forward(test_data[:, 1:])  
#     # Get output from neural network
#     test_points = neural_network.output  

#     # Fit KNN classifier on training data
#     # Create KNN classifier
#     knn_classifier = KNeighborsClassifier(n_neighbors=10)  
#     # Train KNN classifier on training data
#     knn_classifier.fit(train_points, train_data[:, 0])  

#     # Predict labels for test data using trained KNN classifier
#     predicted_labels = knn_classifier.predict(test_points)  

#     # Calculate accuracy and F1 score
#     accuracy = metrics.accuracy_score(test_data[:, 0], predicted_labels)  
#     # Calculate F1 score
#     f1_score = metrics.f1_score(test_data[:, 0], predicted_labels, average='macro', labels=np.unique(predicted_labels))  

#     # Visualize the data with a scatter plot
#     # Get the number of unique classes
#     num_classes = len(np.unique(train_data[:, 0]))  
#     # Create figure and axis for plotting
#     fig, ax = plt.subplots(1, 1, figsize=(6, 6))  
#     # x-coordinates for scatter plot
#     x = train_points[:, 0] 
#     # y-coordinates for scatter plot 
#     y = train_points[:, 1]  
#     # Labels for coloring points
#     labels = train_data[:, 0]  

#     # Choose colormap
#     colormap = plt.cm.jet
#     # Create list of colors from colormap 
#     cmap_list = [colormap(i) for i in range(colormap.N)] 
#     # Create custom colormap 
#     colormap = colormap.from_list('Custom cmap', cmap_list, colormap.N)  

#     # Define boundaries for colormap
#     bounds = np.linspace(0, num_classes, num_classes + 1)  
#     # Normalize colormap
#     norm = plt.Normalize(0, num_classes)  

#     print("IN KNN module")

#     # Create scatter plot
#     scatter = ax.scatter(x, y, c=labels, cmap=colormap, norm=norm) 
#     # Create colorbar 
#     colorbar = plt.colorbar(scatter, spacing='proportional', ticks=bounds) 
#     # Set label for colorbar 
#     colorbar.set_label('Custom colorbar')  
#     plt.show()  

#     # Return accuracy and F1 score
#     return accuracy, f1_score

def compute_accuracy_with_knn_partitioning(neural_network, test_data, train_data):
    '''
    Compute the accuracy of a neural network classifier using KNN partitioning.

    Args:
        neural_network (Neural Network): The trained neural network.
        test_data (numpy.ndarray): Test data with labels.
        train_data (numpy.ndarray): Training data with labels.

    Returns:
        tuple: A tuple containing accuracy and F1 score.
    '''

    # Perform forward pass for training data
    neural_network.forward(train_data[:, 1:])  
    # Get output from neural network
    train_points = neural_network.output  

    # Perform forward pass for test data
    # Exclude labels from input
    neural_network.forward(test_data[:, 1:])  
    # Get output from neural network
    test_points = neural_network.output  

    # Fit KNN classifier on training data
    # Create KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=10)  
    # Train KNN classifier on training data
    knn_classifier.fit(train_points, train_data[:, 0])  

    # Predict labels for test data using trained KNN classifier
    predicted_labels = knn_classifier.predict(test_points)  

    # Calculate accuracy and F1 score
    accuracy = metrics.accuracy_score(test_data[:, 0], predicted_labels)  
    # Calculate F1 score
    f1_score = metrics.f1_score(test_data[:, 0], predicted_labels, average='macro', labels=np.unique(predicted_labels))  

    # Visualize the data with a scatter plot
    # Get the number of unique classes
    num_classes = len(np.unique(train_data[:, 0]))  
    # Create figure and axis for plotting
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  
    # x-coordinates for scatter plot
    x = train_points[:, 0] 
    # y-coordinates for scatter plot 
    y = train_points[:, 1]  
    # Labels for coloring points
    labels = train_data[:, 0]  

    # Choose colormap
    colormap = plt.cm.jet
    # Create list of colors from colormap 
    cmap_list = [colormap(i) for i in range(colormap.N)] 
    # Create custom colormap 
    colormap = colormap.from_list('Custom cmap', cmap_list, colormap.N)  

    # Define boundaries for colormap
    bounds = np.linspace(0, num_classes, num_classes + 1)  
    # Normalize colormap
    norm = plt.Normalize(0, num_classes)  

    # Create scatter plot
    scatter = ax.scatter(x, y, c=labels, cmap=colormap, norm=norm) 
    # Create colorbar 
    colorbar = plt.colorbar(scatter, spacing='proportional', ticks=bounds) 
    # Set label for colorbar 
    colorbar.set_label('Custom colorbar')  
    
    # Display the plot in Streamlit
    st.pyplot(fig)

    # Return accuracy and F1 score
    return accuracy, f1_score
