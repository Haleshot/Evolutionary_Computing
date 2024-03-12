import streamlit as st

st.set_page_config(
    page_title="Particle Swarm Optimization (PSO) Implementation 🧑‍💻",
    layout="wide"
)

st.markdown(
    """

    ## Overview
    This Python file contains the implementation of the Particle Swarm Optimization (PSO) algorithm. PSO is a population-based optimization technique inspired by the social behavior of bird flocking or fish schooling. It is commonly used to find the optimal solution for optimization problems.

    ## Contents
    The file consists of the following components:

    1. **Particle Class Definition**: Defines the Particle class, which represents a single particle in the PSO algorithm. Each particle contains information about its position, velocity, and personal and global best solutions.

    2. **Swarm Class Definition**: Defines the Swarm class, which represents the entire population of particles in the PSO algorithm. It includes methods for initializing the swarm, updating particle positions and velocities, and optimizing the target function.

    3. **Target Function**: Defines the target function to be optimized by the PSO algorithm. This function evaluates the fitness of a particle's position in the search space.

    4. **Main Function**: The main function creates an instance of the Swarm class, initializes the swarm with random particle positions and velocities, and optimizes the target function using the PSO algorithm.

    ## Usage
    To use the PSO algorithm for optimization tasks, follow these steps:

    1. Import the Swarm class from the provided Python file into your project.
    ```python
    import copy 
    import random
    import numpy as np
    from scipy import stats
    from sklearn.neighbors import NearestNeighbors
    ```

    2. Define the target function that you want to optimize.

    3. Create an instance of the Swarm class and specify the number of particles and other parameters.

    4. Initialize the swarm with random particle positions and velocities.

    5. Optimize the target function using the `optimize` method of the Swarm class.

    6. Retrieve the best solution found by the PSO algorithm from the swarm.

    ## Source code explanation
    ---

    The `__init__` method initializes a Particle object with random weights and biases for a neural network. It accepts input data (`x`) and output labels (`y`) as arguments, both defaulting to empty lists.

    - **Inputs:**
        - `x` (list): Input data for the neural network.
        - `y` (list): Output labels for the input data.

    - **Functionality:**
        - Randomly initializes weights and biases for the neural network's hidden and output layers within the range [-4, 4].
        - Stores the fitness value for each neural network.
        - Stores the output of the neural network in the reduced dimensionality partition space.
        - Uses a discrimination weight (`alpha`) for training.
        - If input data and output labels are provided, calculates the fraction of data elements belonging to each class (`weight_class`) and initializes the fitness value.

    ```python
    class Particle:
    '''
    Each individual represents a neural network. We use PSO to train the weights for the neural network.
    '''

    def __init__(self, x=[], y=[]):
        '''
        Initializes a Particle object with random weights and biases.

        Args:
            x (list): Input data for the neural network. Defaults to an empty list.
            y (list): Output labels for the input data. Defaults to an empty list.
        '''
        # Initial weights are set between -4 and +4
        # Refer R. Eberhart and J. Kennedy, A new optimizer using particle swarm theory
        weight_initial_min = -4
        weight_initial_max = 4

        # Initialize weights for the hidden layer
        self.w1 = (weight_initial_max - weight_initial_min) * np.random.random_sample(size=(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES)) + weight_initial_min

        # Initialize weights for the output layer
        self.w2 = (weight_initial_max - weight_initial_min) * np.random.random_sample(size=(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES)) + weight_initial_min

        # Initialize biases for the hidden layer
        self.b1 = (weight_initial_max - weight_initial_min) * np.random.random_sample(size=(1, NUMBER_OF_HIDDEN_NODES)) + weight_initial_min

        # Initialize biases for the output layer
        self.b2 = (weight_initial_max - weight_initial_min) * np.random.random_sample(size=(1, NUMBER_OF_OUTPUT_NODES)) + weight_initial_min

        # Stores fitness value for each neural network
        self.fitness = None

        # Stores output of the neural network in the reduced dimensionality partition space
        self.output = None

        # Discrimination weight
        self.alpha = ALPHA

        if x and y:
            # Weight class contains the fraction of the data elements belonging to each class of the dataset
            self.weight_class = self.frac_class_wt(y)
            
            # Initialize fitness
            self.calc_fitness(x, y)

    ```
    ---
    The `frac_class_wt` function calculates the fraction of each class in the dataset based on the provided output labels.

    - **Inputs:**
        - `arr` (array-like): An array containing the output labels of the dataset.

    - **Returns:**
        - `frac_arr` (list): A list containing the fraction of each class in the dataset.

    - **Functionality:**
        - Initializes a list `frac_arr` to store the fraction of each class.
        - Iterates through the output labels and counts the occurrences of each class.
        - Calculates the fraction of each class by dividing the count by the total number of elements in `arr`.
        - Returns `frac_arr` containing the fraction of each class in the dataset.

    ```python
    def frac_class_wt(self, arr):
    '''
    Computes the fraction of each class in the dataset.

    Args:
        arr (list): The list of output labels.

    Returns:
        list: A list containing the fraction of each class in the dataset.
    '''        
    frac_arr = [0 for i in range(CLASS_NUM)]

    for j in arr:
        class_num = int(j) - 1
        frac_arr[class_num] += 1

    for i in range(len(frac_arr)):
        frac_arr[i] = frac_arr[i] / float(arr.size)
    
    return frac_arr
    ```

    ---
    The `forward` method computes the output of the neural network using the specified activation function. It takes input data (`inp_x`) and an optional activation function (`activation`), defaulting to "sigmoid".

    - **Inputs:**
        - `inp_x` (array-like): Input data for the neural network.
        - `activation` (str, optional): Activation function to be used. Defaults to "sigmoid".

    - **Functionality:**
        - Applies the specified activation function to the weighted sum of inputs for the hidden layer (`z1`) and final layer (`z2`).
        - Computes the output of the neural network (`a2`) using the activation function.
        - Sets the `output` attribute of the `Particle` object to the computed output.


    ```python
    def forward(self, inp_x, activation="sigmoid"):
    '''
    Computes the output of the neural network.

    Args:
        inp_x (array-like): Input data for the neural network.
        activation (str, optional): Activation function to be used. Defaults to "sigmoid".

    Returns:
        None
    '''
    # Specifies activation function (not yet implemented)
    if activation == "sigmoid":
        activation = sigmoid
    elif activation == "tanh":
        activation = tanh
    elif activation == "relu":
        activation = relu
    else:
        raise Exception('Non-supported activation function')
    
    # Activation of hidden layer
    z1 = np.dot(inp_x, self.w1) + self.b1
    a1 = activation(z1)
    
    # Activation (output) of final layer
    z2 = np.dot(a1, self.w2) + self.b2
    a2 = activation(z2)

    # Set the output of the neural network
    self.output = a2

    ```

    ---
    The `calc_fitness` method calculates the fitness of each neural network using a similarity measure specified in the paper.

    - **Inputs:**
        - `inp_x` (array-like): Input data for the neural network.
        - `out_y` (array-like): Output labels for the input data.

    - **Functionality:**
        - Computes the output of the neural network in reduced dimensionality space.
        - Normalizes the output using z-score normalization.
        - Constrains the normalized points within a hypersphere of radius 1.
        - Calculates a similarity matrix to measure the similarity between every pair of records in the dataset.
        - Finds the nearest neighbors for each data point in the output space.
        - Computes the fitness value using equation 6 from the paper, considering the similarity between elements of the same class and different classes.
        - Stores the computed fitness value in the `fitness` attribute of the `Particle` object.


    ```python
    def calc_fitness(self, inp_x, out_y):
    '''
    Calculate the fitness of each neural network using a similarity measure described in the paper.

    Args:
        inp_x (array-like): Input data for the neural network.
        out_y (array-like): Output labels for the input data.

    Returns:
        float: Fitness value calculated for the neural network.
    '''
    n = len(inp_x)

    # Run through the neural network and give output in reduced dimensionality space
    self.forward(inp_x)

    # Apply z-score function to normalize the output
    self.output = stats.zscore(self.output)

    h = np.zeros((n, NUMBER_OF_OUTPUT_NODES))

    # Normalize points constrained in hyperspace
    # Constrain the normalized points into a hypersphere of radius 1
    for i in range(n):
        x_dist = np.linalg.norm(self.output[i])
        numerator = 1 - np.exp(-(x_dist / 2))
        denominator = x_dist * (1 + np.exp(-(x_dist / 2)))
        h[i] = self.output[i] * (numerator / denominator)

    self.output = h

    # Initialize similarity matrix
    similarity_matrix = np.zeros((n, n))

    # Calculate similarity between every two records in the dataset
    for i in range(n):
        for j in range(i, n):
            similarity = 2 - (np.linalg.norm(h[i] - h[j]))
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    # Get the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=NEAREST_NEIGHBORS).fit(self.output)
    _distances, indices = nbrs.kneighbors(self.output)

    # Calculate fitness as per equation 6 in the paper
    f = 0
    for i in range(n):
        f_temp = 0
        for j in indices[i]:
            if out_y[i] == out_y[j]:
                # Similarity for elements of the same class
                f_temp += similarity_matrix[i][j]
            else:
                # Similarity for elements of different classes
                f_temp += self.alpha * similarity_matrix[i][j]

        # Index for the weight_class
        index = int(out_y[i]) - 1
        f += self.weight_class[index] * f_temp

    self.fitness = f
    return f

    ```

    ---
    The `kmeans_eval` method calculates the fitness of each neural network using a similarity measure described in the paper. It takes input data (`inp_x`) for the neural network and normalizes the output in reduced dimensionality space. Then, it constrains the normalized points within a hypersphere of radius 1 and updates the output accordingly.

    - **Inputs:**
        - `inp_x` (array-like): Input data for the neural network.

    - **Functionality:**
        - Runs the neural network to obtain output in reduced dimensionality space.
        - Performs z-score normalization on the output.
        - Constrains the normalized points within a hypersphere of radius 1 to ensure they lie within a defined space.


    ```python
    def kmeans_eval(self, inp_x):
    '''
    Calculate the fitness of each neural network using a similarity measure given in the paper.
    
    Args:
        inp_x (array-like): Input data for the neural network.
        
    Returns:
        None
    '''
    n = len(inp_x)

    # Run through the neural network and give output in reduced dimensionality space
    self.forward(inp_x)

    # Z-score normalization for output
    self.output = stats.zscore(self.output)

    # Initialize array for normalized points constrained in hyperspace
    h = np.zeros((n, NUMBER_OF_OUTPUT_NODES))

    # Normalize points constrained in hyperspace (within a hypersphere of radius 1)
    for i in range(n):
        x_dist = np.linalg.norm(self.output[i])
        numerator = 1 - np.exp(-(x_dist / 2))
        denominator = x_dist * (1 + np.exp(-(x_dist / 2)))
        h[i] = self.output[i] * (numerator / denominator)

    self.output = h

    ```

    ---
    - **Inputs:**
        - `x` (list): Input data for the neural network. Defaults to an empty list.
        - `y` (list): Output labels for the input data. Defaults to an empty list.

    - **Functionality:**
        - Initializes a `Vel` object with zero-initialized weights and biases for the neural network's hidden and output layers.
        - The weight matrices (`w1` and `w2`) are initialized as zero matrices.
        - The bias vectors (`b1` and `b2`) are initialized as zero vectors.
        - If input data and output labels are provided, they can be used to initialize the object.

    ```python
    class Vel:
    '''
    Represents the velocity of particles in the particle swarm optimization.

    Attributes:
        w1 (ndarray): Weight matrix for the hidden layer.
        w2 (ndarray): Weight matrix for the output layer.
        b1 (ndarray): Bias vector for the hidden layer.
        b2 (ndarray): Bias vector for the output layer.
    '''

    def __init__(self, x=[], y=[]):
        '''
        Initializes the Vel object with zero-initialized weights and biases.

        Args:
            x (list): Input data for the neural network. Defaults to an empty list.
            y (list): Output labels for the input data. Defaults to an empty list.
        '''
        # Initial weights are set to zeros
        self.w1 = np.zeros((NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES))
        self.w2 = np.zeros((NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES))

        # Initialize biases to zeros
        self.b1 = np.zeros((1, NUMBER_OF_HIDDEN_NODES))
        self.b2 = np.zeros((1, NUMBER_OF_OUTPUT_NODES))

    ```


"""
)