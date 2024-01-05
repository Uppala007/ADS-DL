Diabetic Retinopathy Classifier
This repository contains the code for a Convolutional Neural Network (CNN) model designed to classify images for the presence of Diabetic Retinopathy, a diabetes-related eye condition. The model is implemented using TensorFlow and Keras.

Prerequisites
To run this code, you need the following installed:

Python 3.x
TensorFlow 2.x
Keras
NumPy
Matplotlib
Seaborn
Scikit-learn
GPU Support
The code is configured to check for GPU availability. If a GPU is available, TensorFlow will use it to accelerate model training. If not, the training will default to the CPU.

Dataset
The dataset should be organized into separate directories for training, validation, and testing. Each directory should contain subdirectories for each class. Update the base_dir, train_dir, valid_dir, and test_dir paths in the script to point to your dataset location.

Image Data Generators
The ImageDataGenerator class is used to augment the training images and to rescale the validation and test images.

Model Architecture
The model is a Sequential CNN with convolutional, max-pooling, flatten, dense, and dropout layers. The output layer uses sigmoid activation for binary classification.

Training
The model is trained using binary cross-entropy loss and the Adam optimizer. Training and validation accuracies and losses are plotted after training.

Saving and Loading the Model
After training, the model is saved as an HDF5 file. It can be loaded later for inference or further training using TensorFlow's load_model function.

Model Evaluation
The script includes code to evaluate the model on a test dataset. It generates a classification report and confusion matrix to assess the model's performance.

Usage
To use this code, clone this repository, install the prerequisites, and run the script. Ensure your dataset path is correctly set in the script.

Contributing
Contributions to improve this project are welcome. Please fork the repository and create a pull request with your enhancements.

License
This project is open-source and available under the MIT License.