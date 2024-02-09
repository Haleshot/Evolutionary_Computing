import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Introduction 👋")

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    **In the realm of supervised classification**, artificial neural networks emerge as powerful tools, capable of approximating diverse nonlinear functions for effective pattern recognition. However, the conventional approach of fixed centroids in partition space limits the flexibility of neural networks during optimization, impacting decision boundary adaptability.

    ---

    ### Our Solution

    Our solution integrates **Particle Swarm Optimization (PSO)** and **k-Nearest Neighbors (kNN)** to enhance neural network classifiers. This hybrid approach leverages PSO for weight optimization, refining the neural network's ability to discriminate and classify patterns effectively.

    ---

    **NNP employs evolutionary computation to optimize neural-network parameters**, ensuring a global search for the best neural network and partition space. NNP eliminates fixed centroids, allowing for flexible decision boundaries between classes in partition space, overcoming the limitations of traditional methods.

    ---

    ### Neural Network Architecture

    The neural network architecture dynamically adjusts its input layer size based on the features in the dataset. It includes a hidden layer with a fixed number of nodes and an output layer corresponding to the dimensions of the reduced dimensionality partition space.

    ---

    ### Fitness Evaluation

    Fitness evaluation incorporates a novel similarity measure, emphasizing both intra-class and inter-class similarities in the reduced dimensionality space. PSO optimizes the neural network weights to maximize this fitness, enhancing discrimination and pattern recognition.

    ---

    ### Performance Evaluation

    After PSO optimization, the neural network's performance is evaluated using k-Nearest Neighbors (kNN) classification. Accuracy and F1 score are calculated, providing insights into the classifier's effectiveness on both training and testing sets.

    ---

    **In conclusion**, our Nearest Neighbor Partitioning (NNP) method revolutionizes neural-network classifiers through a novel partition space mapping approach. NNP excels in creating clear decision boundaries between classes, eliminating the need for centroids. Across seventeen datasets, NNP outperforms seven alternative methods in accuracy and average f-measure, particularly on imbalanced data. Its effectiveness as a dimension-reduction technique for data visualization is demonstrated by consistent accuracy across dimensions. Future research will explore NNP's adaptability to different neural network structures and potential integration with regularization strategies for enhanced generalizability. NNP's flexibility extends to compatibility with various classifiers, enhancing its applicability in diverse classification tasks.

"""
)