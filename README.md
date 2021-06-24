# DL_Course_Project
Project repository for Deep learning course project assignment
# Goal - Using StyleGAN2-ADA to improve classification accuracy on small datasets 
It is known that to train a good deep learning image classifier, it is required to have a sufficiently large training dataset. But what if the training data size is limited? In this project we will attempt to remedy this problem by utilizing the pytorch version of StyleGAN2-ADA (https://github.com/NVlabs/stylegan2-ada-pytorch).

The whole project is based on Pytorch.
# StyleGAN2-ADA
StyleGAN2-ADA is a GAN which learns to create new images that are similar to those in the training set. For example, the creators of this model trained it on real human faces and the resulting images generated can be seen here: https://thispersondoesnotexist.com/ - each visit to the webpage generates a new face which doesnt actually exist.
The ADA suffix stands for 'adaptive discriminator augmentation' which means that a special augmentation pipeline is applied to the data to help stabilize the GAN in limited training data regimes. This characteristic is very beneficial to our cause, as we will be using the GAN to artificially enlarge the small dataset available for training our classifier.
# Dataset
In this project we use a small dataset of 2 classes (Ice-cream and Waffles) created by sapa16. Link: https://www.kaggle.com/sapal6/waffles-or-icecream.
The dataset contains 343 samples for Ice-cream and 355 samples for Waffles.

We divided the first 300 images of both classes for the train set, and the remaining ~50 for the test set.

## GAN train set enlargement
Before we can train the models, we need to generate the artificial train data for the second classifier. W
# Project goals
We show that higher classification accuracy can be achieved if we first use the StyleGAN2-ADA to create more artificial data and train the classifier on it, than to train the classifier only on the original data.

To do so we've conducted two experiments on the same standard classifier architecture:




