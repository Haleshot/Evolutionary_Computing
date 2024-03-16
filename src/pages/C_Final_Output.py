# import libraries
import streamlit as st
import numpy as np
from sklearn.model_selection import KFold
import random
# from .3_PSO import Swarm
# from .2_KNN import accuracy



st.markdown(
    """

    ## Overview
    This Python file contains the implementation of the final code for the project. It includes the integration of the KNN algorithm with Particle Swarm Optimization (PSO) for classification tasks. The final code encompasses data loading, preprocessing, model training, and evaluation steps.

    ## Contents
    The file consists of the following components:

    1. **Data Loading and Preprocessing**: Includes functions to load dataset files, perform data normalization, and prepare data for model training.

    2. **Particle Swarm Optimization (PSO)**: Defines the PSO algorithm implementation for optimizing the weights of the KNN classifier. PSO is used to search for the optimal weights that maximize the classification accuracy.

    3. **K-Nearest Neighbors (KNN) Algorithm**: Implements the KNN algorithm for classification. It includes functions for model training, prediction, and evaluation.

    4. **Main Functionality**: The main function orchestrates the entire workflow, including data loading, PSO optimization, KNN model training, and evaluation.

    ## Usage
    To utilize the final code for classification tasks, follow these steps:

    1. Import the necessary functions and classes from the provided Python file into your project.
    ```python
    import numpy as np
    from sklearn.model_selection import KFold
    import random
    from pso import Swarm # TODO: Custom library to be defined
    from knn_code import accuracy # TODO: Custom library to be defined
    ```

    2. Prepare your dataset or use one of the provided datasets for classification.

    3. Invoke the main function to execute the entire workflow, including data loading, PSO optimization, KNN model training, and evaluation.

    4. Evaluate the performance of the trained KNN model using metrics such as accuracy, F-score, etc.
    
    ## Source code explanation
    ```python

    FILENAME = "glass+identification\glass.data" # TODO: Keep glass as default, but make it so that it depends on user selection from dataset dropdown
    ```
    ---
    The `run` function executes the algorithm for each test and training set pair. It initializes a Swarm object, initializes the swarm with the training dataset, optimizes the swarm, and calculates the accuracy of the optimized solution.

    **Parameters:**
    - `test` (array-like): The test dataset.
    - `train` (array-like): The training dataset.

    **Returns:**
    - `float`: The accuracy of the algorithm.


    ```python
    def run(test, train):
        '''
        Run the algorithm for each test and training set pair.

        Args:
            test (array-like): The test dataset.
            train (array-like): The training dataset.

        Returns:
            float: The accuracy of the algorithm.
        '''
        # Initialize a Swarm object
        s = Swarm()
        
        # Initialize the swarm with the training dataset
        s.initialize_swarm(train)
        
        # Optimize the swarm
        ans = s.omptimize()  # Typo: should be optimize() instead of omptimize()
    
        # Calculate the accuracy of the optimized solution
        return accuracy(ans, test, train)
    ```
    ---
    The `normalize` function performs Min-Max normalization of the features before running the algorithm. It takes a dataset as input and returns the normalized dataset. The function normalizes the features between 0 and 1, handles class labels based on the specified class flag, and concatenates the class labels with the normalized features.

    **Parameters:**
    - `dataset` (array-like): The dataset to be normalized.

    **Returns:**
    - `array-like`: The normalized dataset.

    
    ```python
    def normalize(dataset):
        '''
        Minmax Normalization of the features before running the algorithm.

        Args:
            dataset (array-like): The dataset to be normalized.

        Returns:
            array-like: The normalized dataset.
        '''
        # Normalize the features between 0 and 1
        h = 1
        l = 0
        mins = np.min(dataset, axis=0)
        maxz = np.max(dataset, axis=0)
        
        rng = maxz - mins
        # Result containing the normalized feature
        res = h - (((h - l) * (maxz - dataset)) / rng)

        # Set the class_flag to -1 if the class label is the last column
        # Set class_flag to 0 if class label is the first column
        class_flag = int(input("Enter class flag"))

        # Remove the class column and add back the unnormalized class label
        # Class labels should not be normalized
        if class_flag == 0:
            res = res[:, 1:]
            dataset = dataset[:, 0]
            dataset = dataset.reshape(-1, 1)
        elif class_flag == -1:
            res = res[:, :-1]
            dataset = dataset[:, -1]
            dataset = dataset.reshape(-1, 1)

        # Concatenate the class labels with the normalized features along the column axis
        out = np.concatenate((dataset, res), axis=1)
        
        return out
    ```
    ---

    The `loadfile` function loads the data from a file and normalizes it. It takes the path to the file containing the dataset as input and returns the normalized dataset as a NumPy array. The function excludes the header row (if any), shuffles the dataset to introduce randomness, and normalizes the dataset using the `normalize` function.

    **Parameters:**
    - `filename` (str): The path to the file containing the dataset.

    **Returns:**
    - `numpy.ndarray`: The normalized dataset.


    ```python
    def loadfile(filename):
        '''
        Load the data from the file and normalize it

        Args:
            filename (str): The path to the file containing the dataset

        Returns:
            numpy.ndarray: The normalized dataset
        '''
        # Load the dataset from the file
        dataset = np.genfromtxt(filename, delimiter=',')

        # Exclude the header row (if any)
        dataset = dataset[1:]

        # Shuffle the dataset to introduce randomness
        np.random.shuffle(dataset)

        # Normalize the dataset using the normalize function
        dataset = normalize(dataset)

        return dataset
    ```
    ---
    The `kfold` function performs k-fold cross-validation to test for accuracy. It takes a dataset as input and splits it into 10 folds. For each fold, the function runs the algorithm for each test and training set pair, calculates and prints the accuracy and F-score, and stores the results. Finally, it calculates and prints the average accuracy and F-score across all folds.

    **Parameters:**
    - `dataset` (numpy.ndarray): The dataset to be used for k-fold validation.

    **Returns:**
    - `None`

    ```python
    def kfold(dataset):
        '''
        Perform k-fold cross-validation to test for accuracy

        Args:
            dataset (numpy.ndarray): The dataset to be used for k-fold validation

        Returns:
            None
        '''
        # Initialize k-fold object with 10 splits
        kf = KFold(n_splits=10)
        kf.get_n_splits(dataset)
        
        # Lists to store accuracy and F-score for each fold
        avg_acc = []
        avg_fscr = []
    
        # Iterate over each fold
        for train_ind, test_ind in kf.split(dataset):
            # Split dataset into train and test sets
            train, test = dataset[train_ind], dataset[test_ind]
            
            # Run the algorithm for each test and training set pair
            acc, fscr = run(test, train)
            
            # Print accuracy and F-score for the current fold
            print("Accuracy ", acc, " F-score ", fscr)
            
            # Store accuracy and F-score for the current fold
            avg_acc.append(acc)
            avg_fscr.append(fscr)
    
        # Calculate average accuracy and F-score across all folds
        avg_acc_ans = sum(avg_acc) / len(avg_acc)
        avg_fscore_ans = sum(avg_fscr) / len(avg_fscr)
        
        # Print average accuracy and F-score
        print("Average accuracy ", avg_acc_ans)
        print("Average fscore ", avg_fscore_ans)

    ```
    ---
    The `main` function serves as the entry point for the algorithm. It loads the dataset from the specified file using the `loadfile` function, performs k-fold cross-validation to test accuracy using the `kfold` function, and prints the filename before executing the validation.

    **Returns:**
    - `None`


    ```python
    def main():
        '''
        Start the algorithm and use k-fold validation to test accuracy

        Returns:
            None
        '''
        # Load the dataset from the file
        dataset = loadfile(FILENAME)
        
        # Print the filename
        print(FILENAME)
        
        # Perform k-fold cross-validation to test accuracy
        kfold(dataset)
    ```
    ---

    ```python
    # Invoke the main function to execute the workflow
    main()
    ```

    """
)




############################################## CODE IMPLEMENTATION ##############################################################
import numpy as np
from sklearn.model_selection import KFold
import random
# from .3_PSO import Swarm # TODO: Custom library to be defined
# from .2_KNN import accuracy # TODO: Custom library to be defined
from pages.B_PSO import Swarm
from A_KNN import accuracy




FILENAME = "glass+identification\glass.data" # TODO: Keep glass as default, but make it so that it depends on user selection from dataset dropdown



def run(test, train):
    '''
    Run the algorithm for each test and training set pair.

    Args:
        test (array-like): The test dataset.
        train (array-like): The training dataset.

    Returns:
        float: The accuracy of the algorithm.
    '''
    # Initialize a Swarm object
    s = Swarm()

    # Initialize the swarm with the training dataset
    s.initialize_swarm(train)

    # Optimize the swarm
    ans = s.omptimize()  # Typo: should be optimize() instead of omptimize()

    # Calculate the accuracy of the optimized solution
    return accuracy(ans, test, train)




def normalize(dataset):
    '''
    Minmax Normalization of the features before running the algorithm.

    Args:
        dataset (array-like): The dataset to be normalized.

    Returns:
        array-like: The normalized dataset.
    '''
    # Normalize the features between 0 and 1
    h = 1
    l = 0
    mins = np.min(dataset, axis=0)
    maxz = np.max(dataset, axis=0)

    rng = maxz - mins
    # Result containing the normalized feature
    res = h - (((h - l) * (maxz - dataset)) / rng)

    # Set the class_flag to -1 if the class label is the last column
    # Set class_flag to 0 if class label is the first column
    class_flag = int(input("Enter class flag"))

    # Remove the class column and add back the unnormalized class label
    # Class labels should not be normalized
    if class_flag == 0:
        res = res[:, 1:]
        dataset = dataset[:, 0]
        dataset = dataset.reshape(-1, 1)
    elif class_flag == -1:
        res = res[:, :-1]
        dataset = dataset[:, -1]
        dataset = dataset.reshape(-1, 1)

    # Concatenate the class labels with the normalized features along the column axis
    out = np.concatenate((dataset, res), axis=1)

    return out


def loadfile(filename):
    '''
    Load the data from the file and normalize it

    Args:
        filename (str): The path to the file containing the dataset

    Returns:
        numpy.ndarray: The normalized dataset
    '''
    # Load the dataset from the file
    dataset = np.genfromtxt(filename, delimiter=',')

    # Exclude the header row (if any)
    dataset = dataset[1:]

    # Shuffle the dataset to introduce randomness
    np.random.shuffle(dataset)

    # Normalize the dataset using the normalize function
    dataset = normalize(dataset)

    return dataset



def kfold(dataset):
    '''
    Perform k-fold cross-validation to test for accuracy

    Args:
        dataset (numpy.ndarray): The dataset to be used for k-fold validation

    Returns:
        None
    '''
    # Initialize k-fold object with 10 splits
    kf = KFold(n_splits=10)
    kf.get_n_splits(dataset)

    # Lists to store accuracy and F-score for each fold
    avg_acc = []
    avg_fscr = []

    # Iterate over each fold
    for train_ind, test_ind in kf.split(dataset):
        # Split dataset into train and test sets
        train, test = dataset[train_ind], dataset[test_ind]

        # Run the algorithm for each test and training set pair
        acc, fscr = run(test, train)

        # Print accuracy and F-score for the current fold
        print("Accuracy ", acc, " F-score ", fscr)

        # Store accuracy and F-score for the current fold
        avg_acc.append(acc)
        avg_fscr.append(fscr)

    # Calculate average accuracy and F-score across all folds
    avg_acc_ans = sum(avg_acc) / len(avg_acc)
    avg_fscore_ans = sum(avg_fscr) / len(avg_fscr)

    # Print average accuracy and F-score
    print("Average accuracy ", avg_acc_ans)
    print("Average fscore ", avg_fscore_ans)


def main():
    '''
    Start the algorithm and use k-fold validation to test accuracy

    Returns:
        None
    '''
    # Load the dataset from the file
    dataset = loadfile(FILENAME)

    # Print the filename
    print(FILENAME)

    # Perform k-fold cross-validation to test accuracy
    kfold(dataset)

# Invoke the main function to execute the workflow
main()