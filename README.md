3D Shape Classification with PointNet
This project implements the PointNet deep learning architecture to classify 3D shapes using the ModelNet10 dataset. The model takes raw point cloud data as input and outputs a classification label for the 3D object.

Objective
The primary goal of this project is to build and train a neural network capable of understanding and classifying 3D objects from their point cloud representations. This demonstrates the effectiveness of the PointNet architecture in handling unordered point sets, a common type of 3D geometric data.

Method
Architecture
The model is based on the PointNet architecture, which is specifically designed to work with unordered point sets. Its key features include:

Permutation Invariance: It uses a symmetric function (max pooling) to aggregate features from individual points, making the model invariant to the input order of points.

Shared Multi-Layer Perceptrons (MLPs): The same MLP is applied to each point independently to learn spatial encodings. This is implemented using 1D convolutional layers.

Global Feature Extraction: After processing individual points, a max-pooling layer aggregates the point features into a single global feature vector that represents the entire shape.

Classification Head: A final set of fully connected layers takes this global feature vector and performs the classification task.

Dataset
The project uses the ModelNet10 dataset, which is a subset of the larger ModelNet dataset. It contains 10 categories of 3D CAD models. For this project, the data is pre-processed into point clouds, with each object represented by 2048 points.

Data Augmentation
To improve the model's robustness and prevent overfitting, the following data augmentation techniques are applied to the training data in real-time:

Random Rotation: Randomly rotates the point cloud along the vertical (z) axis.

Random Jitter: Adds a small amount of random noise (jitter) to each point's coordinates.

A sample visualization of a batch of point clouds from the dataset is shown below:

Setup Process
Prerequisites
Python 3.8+

Jupyter Notebook or JupyterLab

PyTorch

NumPy

Matplotlib

Installation
Clone the repository:

git clone <repository-url>
cd 3d-shape-classification

Install dependencies:
It is recommended to use a virtual environment.

pip install torch torchvision numpy matplotlib notebook

Prepare the dataset:

Create a data/ directory in the project root.

Download the ModelNet10 point cloud data (points_train.npy, labels_train.npy, points_test.npy, labels_test.npy).

Place the .npy files inside the data/ directory. Your project structure should look like this:

.
├── data/
│   ├── points_train.npy
│   ├── labels_train.npy
│   ├── points_test.npy
│   └── labels_test.npy
├── pointnet_classifier.ipynb
└── README.md

Run the notebook:
Launch Jupyter and open the pointnet_classifier.ipynb file.

jupyter notebook

Once the notebook is open, you can run all the cells to train the model. The notebook will periodically print the loss and accuracy, and it will save the model with the best validation accuracy as best_model.pth.

Results
The model is trained for 50 epochs using the Adam optimizer and Negative Log-Likelihood Loss (NLLLoss). The training process involves monitoring both training and validation accuracy to select the best-performing model.

A typical output during training will look like this:

Using device: cuda
Loaded train data: (3991, 10000, 3), labels: (3991, 1)
Loaded test data: (908, 10000, 3), labels: (908, 1)
Training samples: 3991
Testing samples: 908
Saved visualization to visualize.jpg
--- Epoch 1/50 ---
[Batch: 10/125] Loss: 2.0524
[Batch: 20/125] Loss: 1.7011
...
Epoch 1 ==> Train Loss: 1.6312, Train Acc: 42.50%
Epoch 1 ==> Val Loss: 1.4870, Val Acc: 51.54%
Validation accuracy improved. Saved best model!
--- Epoch 2/50 ---
...

After training, the model achieves a high classification accuracy on the test set, demonstrating its ability to effectively learn distinguishing features from raw 3D point cloud data. The final best_model.pth file can be used for inference on new point cloud data.
