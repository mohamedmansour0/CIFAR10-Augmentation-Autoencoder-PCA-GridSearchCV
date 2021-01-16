# CIFAR10-Autoencoder-PCA-GridSearch
Investigating various techniques to enhance image  classification

# Introduction:
Enhancing the efficiency of classifying a dataset is crucial. Different techniques are used to
achieve this goal, but which one is more effective? In this project an investigation of using
different techniques on one of the most popular datasets which is CIFAR10 is done.
This investigation aims to compare those different techniques in terms of training and
test accuracies, precision, and time consumption. Different techniques that are data
augmentation, data reduction, and feature extraction are applied to the data to study their
effect. Also, a hyperparameter tuning is done to determine the variables that give best results.

# Dataset
The dataset that are used in this project are CIFAR10.
CIFAR10 dataset contains 60000 32x32x3 images labeled in 10 classes that are automobile,
truck, cat, frog, airplane, horse, deer, bird, ship, and dog. These classes are mutually
exclusive and balanced, that is every class has 6000 images. It is splitting by this way 50000
images for training and 10000 images for testing.

# Methodology
The first step of this project is to load both datasets and normalize them. Then, we applied the
following techniques:
  # Augmentation
  We applied at least one of the following changes randomly for each image.
  - Rotation=30°.
  - Horizontal flip.
  - Width shift range=0.1.
  - Height shift range=0.1.
  - Zoom range=0.3.
  After augmentation, we have three different training outputs for each dataset. Sizes of the
  outputs are as follows:
  
  Original training output: CIFAR10 (50,000x32x32x3).
  Original training output + augmented: CIFAR10 (100,000x32x32x3).
  10% randomly selected of original training output + augmented: CIFAR10 (10,000x32x32x3).
  
  # Data reduction
  Since one of the comparison factors is time consumption, the data dimensions are reduced
  by using Principal Component Analysis. It is expected that the accuracy will decrease
  because part of the data is removed. However, it is expected to take less time to train the data.
  The variance that we decide to hold for CIFAR10 is 0.95. The following
  shows the number of the principal components that holds 0.95 for each one.
 
  Original training output 217
  Original training output + augmented 217
  
  The PCA is applied on the original training output of CIFAR10, and then the
  output is trained by a neural network to compare it to the other techniques.
  
  # Feature extraction
  The Dataset used in this process was the 10% of the randomly selected original training and
  the augmented training. The autoencoder consists of four main layers: the encoder, middle,
  decoder and the output layers. The output layer would produce the feature extracted layer,
  while the bottleneck features were extracted from the middle layer then applied into classifier
  model to ensure their operation, later on those same bottleneck features are omitted to the
  tuning hyperparameters to be compared with other techniques. It is believed that this output
  will perform the least in accuracy and precision, but it would be the fastest in execution since
  it was trained on only 10% of the two given datasets.
  
  # Hyperparameter Tuning
  In the matter of choosing the model hyperparameters we used grid search algorithm to
  determine the best hyperparameters. Since the model have several hyperparameters and with
  limited computation resources, we fixed some of the hyperparameters and chose few variable
  values others. The number of layers and neurons, the kernel filter size, and the batch size
  have various values. Also, we used two different optimizers. This tuning was applied on the
  original datasets and the augmented ones. The selected hyperparameters based on the proper
  tuning were used to build a model and test the performance.
  
# Results

The following shows the results on the CIFAR10

Test Accuracy % Precision %

Time Consumption(S)

Original CIFAR10: Training Accuracy 77.68 % / Test Accuracy 66.45 % / Precision 66 % / Time Consumption 322 sec

Augmented CIFAR10: Training Accuracy 67.25 % / Test Accuracy 64.66 % / Precision 69 % / Time Consumption 342 sec

PCA on the original CIFAR10: Training Accuracy 64.57 % / Test Accuracy 57.06 % /  Precision 57 % /  Time Consumption 32 sec

Feature Extraction on 10% randomly selected of original training output + augmented
Training Accuracy 65.9 % / Test Accuracy 46 % / Precision 47 % /  Time Consumption 7 sec

Regarding CIFAR10, training CNN on the original dataset based on the best hyperparameters
chosen achieves the highest training and testing accuracies while PCA achieves lower time
consumption when it is trained by using neural network on the same dataset. However, both
training and testing accuracies dropped as expected. Also, that happened in the augmented
data, but the precision increased which is an advantage for the augmented data.
  
# Conclusions & Lessons Learned
Increasing the dataset samples doesn’t always lead to accuracy improvement, as noticed in
CIFAR10, but the precision increased by 3%. The PCA isn’t the best option when it comes to
the accuracy of classifying pictures in CIFAR10. However, it takes less time to train,
because it works better with the lower dimensions.
The feature extraction was able to predict the test set with slightly lower
efficiency but achieved the fastest time since it can work with only 10% of the data. The
hyperparameters selection would be better to have different variation based on the dataset.
