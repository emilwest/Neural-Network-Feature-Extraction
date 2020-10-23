![alt text](all_pics/intro.png "Picture source: Liu et. al. 2015: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html ")

# Neural-Network-Feature-Extraction
Image Recognition using a convolutional base from VGG-16, extracting features from final layers as input for LASSO and Random Forest algorithms. 

> Authors: Emil Westin, Hongru Zhai, Christoffer Eriksson (2020)


## Objective 

- Find the best neural network on top of a pretrained convolutional base (VGG 16).
- Investigate the possibility for some ML models to outperform NN, based on its own generated features.
- In our case we will try doing this with a binary image classification problem.
- We tried to classify images of people as Male or Female.

## Data

- 202 599 images
- 40 binary attributes, e.g. male, glasses, beard.
- We used 8000 images, 1 binary class (male)
- 1600 as training data, 400 as validation data and 6000 as test data

## Method

Based on our classification task we used:
- Loss function: Binary crossentropy
- Optimizer: RMSPROP
- Activation function of the last layer: Sigmoid



- Baseline NN model: one dense layer of 20 units
insert pic

- Final NN model:
insert pic


-  Trained different LASSO- and random forest models for
various numbers of features extracted by using 10-fold
cross validation


## Loss and Accuracy levels

insert pic


## Results

insert table


## Discussion

- Using 20 extracted features, our random forest achieved slightly better accuracy on test data than the NN.
- But the results of these models are similar in general, so it’s hard to draw any conclusions.
- The number of features extracted from the NN seem to make little difference in terms of accuracy

- THINGS WE COULD HAVE DONE DIFFERENT:
- Could have used more machine learning methods
- Used different convolutional bases and/or used different data




