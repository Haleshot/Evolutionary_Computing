import streamlit as st

st.set_page_config(
    page_title="Final Code Output 🧑‍💻",
    layout="wide"
)


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

    2. Prepare your dataset or use one of the provided datasets for classification.

    3. Invoke the main function to execute the entire workflow, including data loading, PSO optimization, KNN model training, and evaluation.

    4. Evaluate the performance of the trained KNN model using metrics such as accuracy, F-score, etc.

    ## Example
    ```python
    def main():
        # Logic goes here

    # Invoke the main function to execute the workflow
    main()
    ```

    """
)
