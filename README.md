# Image-Classification-using-CNN-Model

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is built using the MobileNet architecture, which is lightweight and efficient for deployment in real-time applications. The project is implemented on Google Colab and deployed as a web application using Streamlit for interactive use.

Project Overview:
Image classification is a critical task in computer vision with applications in fields like healthcare, security, and e-commerce. In this project, we leverage the MobileNet CNN architecture to build an efficient image classification model that can classify images from the CIFAR-10 dataset into 10 predefined classes.

Key Features:
MobileNet for efficient image classification on limited hardware
Google Colab for model training with GPU acceleration
Streamlit for easy deployment of the trained model as a web application
CIFAR-10 dataset consisting of 60,000 images categorized into 10 classes
Technologies Used
Google Colab: A cloud-based platform that provides GPU/TPU support for training the CNN model.
Keras/TensorFlow: Frameworks used to build, train, and evaluate the CNN model.
Streamlit: Framework for deploying the trained model as an interactive web app for real-time image classification.

Pre-requisites

pip install tensorflow
pip install streamlit
pip install matplotlib
pip install seaborn
pip install scikit-learn


Model Performance
After training, the model achieves an accuracy of 89% on the CIFAR-10 test dataset. The model is evaluated using several metrics such as accuracy, precision, recall, and F1-score.

Future Work
Experiment with transfer learning to enhance model accuracy on larger datasets like ImageNet.
Optimize the model for real-time inference on edge devices using TensorFlow Lite.
Improve the Streamlit app by adding features like batch image uploads or visualizing intermediate features of the CNN.
