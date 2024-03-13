import streamlit as st

st.set_page_config(
    page_title="K-Nearest Neighbor (KNN) Implementation 🧑‍💻",
    layout="wide"
)

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
    The `forward_pass` function executes a forward pass through a neural network model using the given input data, returning the output of the neural network.
    - **Inputs:**
        - nn_model (NeuralNetworkModel): The neural network model to perform the forward pass.
        - data (numpy.ndarray): Input dataset for the neural network.

    - **Functionality:**
        - Performs a forward pass through the neural network model using the provided input data.
        - Extracts features from the input dataset.
        - Returns the output of the neural network after the forward pass.
    ```python
    def forward_pass(nn_model, data):
        '''
        Performs a forward pass in a neural network and returns the output.
    
        Args:
            nn_model (NeuralNetworkModel): The neural network model.
            data (numpy.ndarray): Input dataset.
        
        Returns:
            numpy.ndarray: Output of the neural network after the forward pass.
        '''    
            nn_model.forward(data[:, 1:])
            return nn_model.output
    ```
    
    ---
    The `train_classifier_knn` function trains a k-Nearest Neighbors (kNN) classifier using the provided training points and labels.
    - **Inputs:**
        - nn_model (NeuralNetworkModel): The neural network model to perform the forward pass.
        - data (numpy.ndarray): Input dataset for the neural network.

    - **Functionality:**
        - Performs a forward pass through the neural network model using the provided input data.
        - Extracts features from the input dataset.
        - Returns the output of the neural network after the forward pass.
    
    - **Returns:**
        - numpy.ndarray: Output of the neural network after the forward pass.
    ```python
    def train_classifier_knn(train_points, train_labels, k_neighbors=10):
        '''
        Trains a k-Nearest Neighbors (kNN) classifier using the provided training points and labels.
        
        Args:
            train_points (numpy.ndarray): Data points from the training set.
            train_labels (numpy.ndarray): Labels corresponding to each data point in the training set.
            k_neighbors (int, optional): Number of neighbors for kNN. Defaults to 10.
        
        Returns:
            KNeighborsClassifier: The trained kNN classifier.
        '''    
            knn_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
            knn_classifier.fit(train_points, train_labels)
            return knn_classifier    
    ```
    
    ---
    The `data_visualization` function performs visualization of data points along with their labels, using a scatter plot colored based on class labels, providing insights into the distribution and patterns present in the dataset.
    - **Inputs:**
        - data_points (numpy.ndarray): Data points to be visualized.
        - data_labels (numpy.ndarray): Labels corresponding to each data point.
        - N (int, optional): Number of classes for color mapping. Defaults to 4.

    - **Functionality:**
        - Evaluates the neural network model using k-Nearest Neighbors (kNN) and returns accuracy and F1 score.
        - Visualizes the data points with scatter plot colored based on class labels.
        - Uses a specified colormap ('jet') for color mapping.
        - Generates a colorbar for visual reference of class boundaries.
    ```python
    def data_visualization(data_points, data_labels, N=4):
        '''
        Evaluates the neural network model using k-Nearest Neighbors (kNN) and returns accuracy and F1 score.
    
        Args:
            nn_model (NeuralNetworkModel): The neural network model to be evaluated.
            test_data (numpy.ndarray): Test dataset (including labels).
            train_data (numpy.ndarray): Training dataset (including labels).
            k_neighbors (int, optional): Number of neighbors for kNN. Defaults to 10.
    
        Returns:
            tuple: A tuple containing two elements - accuracy and F1 score.
        '''    
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
    ```
    
    ---
    The `evaluate_nn` function helps to evaluate the neural network model and it gives output as Accuracy and F1 score.
    - **Inputs:**
        - nn_model (NeuralNetworkModel): The neural network model to be evaluated.
        - test_data (numpy.ndarray): Test dataset (including labels).
        - train_data (numpy.ndarray): Training dataset (including labels).
        - k_neighbors (int, optional): Number of neighbors for kNN. Defaults to 10.

    - **Functionality:**
        - Evaluates the neural network model using k-Nearest Neighbors (kNN) by performing forward passes on both training and test datasets.
        - Trains a kNN classifier using the forward pass results of the training data.
        - Predicts labels for the test data using the trained kNN classifier.
        - Calculates accuracy and F1 score based on predicted labels and actual test labels.
        
    - **Returns:**
        - tuple: A tuple containing two elements - accuracy and F1 score.
    ```python
    def evaluate_nn(nn_model, test_data, train_data, k_neighbors=10):
        '''
        Evaluates the neural network model using k-Nearest Neighbors (kNN) and returns accuracy and F1 score.
    
        Args:
            nn_model (NeuralNetworkModel): The neural network model to be evaluated.
            test_data (numpy.ndarray): Test dataset (including labels).
            train_data (numpy.ndarray): Training dataset (including labels).
            k_neighbors (int, optional): Number of neighbors for kNN. Defaults to 10.
        
        Returns:
            tuple: A tuple containing two elements - accuracy and F1 score.
        '''    
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
    ```   
        """)


