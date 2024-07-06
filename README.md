Brain Tumor Detection Using Convolutional Neural Networks

Introduction

Background
Medical imaging plays a crucial role in diagnosing brain tumors, where accurate and timely detection is vital for patient treatment and prognosis. Convolutional Neural Networks (CNNs) have demonstrated significant success in automated image recognition tasks, making them suitable for medical image analysis, including brain tumor detection.

Objective
The objective of this project is to develop and evaluate a CNN model capable of accurately classifying brain MRI images as either containing a tumor or being tumor-free.

 Dataset
The dataset used in this project consists of brain MRI images sourced from Kaggle. The dataset is organized into three main directories:
- TRAIN: Contains training images with subdirectories `yes` and `no` for tumor and non-tumor images, respectively.
- VAL: Contains validation images similarly organized.
- TEST: Contains test images for evaluating the trained model.

Methodology

Data Preprocessing
The images were resized to 150x150 pixels and normalized to the range [0, 1]. 

Model Architecture
The CNN architecture used for this project is as follows:
- Input Layer: 2D Convolutional layer with ReLU activation
- Hidden Layers: MaxPooling layers for downsampling, followed by additional Conv2D layers with increasing filter sizes
- Flatten layer to convert 2D feature maps into a 1D vector
- Dense layers with ReLU activation for feature processing
- Output Layer: Dense layer with sigmoid activation for binary classification (tumor vs. non-tumor)

Training Setup
- Optimizer: Adam optimizer
- Loss Function: Binary Cross-Entropy
- Metrics: Accuracy

Implementation

Environment
The project was implemented using Python 3.8, TensorFlow 2.x, and Keras on MacBook Air M1 2020.

Results

Training Results
The model was trained for 10 epochs on the `TRAIN` dataset, achieving the following performance metrics:

Testing Results
The trained model was evaluated on the `TEST` dataset, achieving an accuracy of 93.8% 

Discussion

Interpretation of Results
The achieved accuracy on the test set demonstrates the effectiveness of the CNN model in accurately classifying brain MRI images. The validation results indicate good generalization capabilities of the model, despite the data augmentation and dropout regularization used during training.
Future Work
Future work could focus on:
- Fine-tuning hyperparameters to improve model performance.
- Investigating more advanced CNN architectures or transfer learning techniques.
- Incorporating additional medical datasets to further validate and generalize the model.

Conclusion

Summary
In conclusion, this project successfully developed a CNN model for brain tumor detection from MRI images. The model achieved promising results in terms of accuracy and loss, demonstrating its potential for real-world applications in medical diagnostics.

Achievements
Key achievements include:
- Implementation of a robust CNN architecture for medical image classification.
- Successful training and evaluation on a curated dataset of brain MRI images.
- Contribution to the field of automated medical image analysis.
