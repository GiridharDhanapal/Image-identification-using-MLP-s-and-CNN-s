# Image-identification-using-MLP-s-and-CNN-s

<img width="593" alt="MLP1" src="https://github.com/GiridharDhanapal/Image-identification-using-MLP-s-and-CNN-s/assets/117945886/889d9fde-f7ff-4676-973c-a6cfd436f283">

This project is about to build different neural network models like Multilayer Perceptron (MLP) networks and Convolutional Neural Networks (CNNs) to identify characters present in the image dataset. This project aims to build different neural network models to solve the EMNIST problem. The EMNIST dataset is an extension of the well-known MNIST dataset, and it includes handwritten character digits. We will explore Multilayer Perceptron (MLP) networks and Convolutional Neural Networks (CNNs) with various techniques to enhance their performance. Below is a comprehensive explanation of the project, including the dataset, model structures, and the rationale behind my design choices.

# Overview

**Details of the dataset**

In this project, we use the EMNIST (Extended MNIST) dataset, specifically the "Balanced" split. This split addresses balance issues in other datasets and provides an equal number of samples per class. The dataset is derived from the NIST Special Database 19 and converted to a 28x28 pixel image format, matching the structure of the MNIST dataset. Train Set has 112,800 samples, Test Set has 18,800 samples and a total of 131,600 samples.

**Visualization of the Dataset**

The images in the EMNIST dataset look similar to the original MNIST dataset. We visualized a few samples from the dataset to understand its structure better.

**Implementation**

<img width="431" alt="mlp3" src="https://github.com/GiridharDhanapal/Image-identification-using-MLP-s-and-CNN-s/assets/117945886/b2760d54-0a7b-47a5-b55e-fbeb67a711dc">

Multilayer Perceptron (MLP) Networks with at least three hidden layers and the number of layers can be changed accordingly.
Convolutional Neural Networks (CNNs) with at least two convolutional layers and the number of layers can be changed accordingly.
For each network model, i have explore various techniques to enhance performance, including:

* Adaptive Learning Rate

* Activation Functions

* Optimizers

* Batch Normalization

* L1 & L2 Regularization

* Dropout

<img width="728" alt="MLP2" src="https://github.com/GiridharDhanapal/Image-identification-using-MLP-s-and-CNN-s/assets/117945886/28ae7ddb-2566-4e5a-ae12-280350a9926c">

# Conclusion

In this project, we successfully built and compared different neural network models for the EMNIST classification problem. Through various techniques and hyperparameter tuning, we were able to enhance the performance of our models. This project provided valuable insights into the design and training of neural networks for image classification tasks.

<img width="383" alt="MLP4" src="https://github.com/GiridharDhanapal/Image-identification-using-MLP-s-and-CNN-s/assets/117945886/dc6ddff2-33a2-4d5b-94d6-79419dde0efe">

# Tech Stack Used in This Project

**Programming Language**

Python: The primary language used for implementing the models and data handling.

**Libraries**

Pandas: Used for data manipulation and analysis.

NumPy: Utilized for numerical operations and handling arrays.

Matplotlib: Employed for plotting and visualizing data and results.

Seaborn: For advanced data visualization, especially for plotting the confusion matrix.

**Deep Learning Frameworks**

PyTorch: The primary deep learning framework used for building, training, and evaluating neural network models.

TensorFlow: Another powerful deep learning framework used for model building and training.

Keras: A high-level API for TensorFlow, used to simplify the process of building and training neural network models.

<img width="413" alt="MLP5" src="https://github.com/GiridharDhanapal/Image-identification-using-MLP-s-and-CNN-s/assets/117945886/310c9a73-6f6f-4fbd-8f2d-6cb06b8b7bce">


