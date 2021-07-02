# DL_Course_Project
Project repository for Deep learning course project assignment.

Students:

Vladimir Kulikov

Daniel Bracha

# Goal - Using StyleGAN2-ADA to improve classification accuracy on small datasets 
It is known that to train a good deep learning image classifier, it is required to have a sufficiently large training dataset. But what if the training data size is limited? In this project we will attempt to remedy this problem by utilizing the pytorch version of StyleGAN2-ADA (https://github.com/NVlabs/stylegan2-ada-pytorch).

The whole project is based on Pytorch.
# StyleGAN2-ADA
StyleGAN2-ADA is a GAN which learns to create new images that are similar to those in the training set. For example, the creators of this model trained it on real human faces and the resulting images generated can be seen here: https://thispersondoesnotexist.com/ - each visit to the webpage generates a new face which doesnt actually exist.
The ADA suffix stands for 'adaptive discriminator augmentation' which means that a special augmentation pipeline is applied to the data to help stabilize the GAN in limited training data regimes. This characteristic is very beneficial to our cause, as we will be using the GAN to artificially enlarge the small dataset available for training our classifier.
# Dataset
In this project we use a small dataset of 2 classes (Ice-cream and Waffles) created by sapal6. Link: https://www.kaggle.com/sapal6/waffles-or-icecream.
The dataset contains 343 samples for Ice-cream and 355 samples for Waffles.

We divided the first 300 images of both classes for the train set, and the remaining ~50 for the test set.

# The Classifiers
Both classifiers used are VGG16 with the last fc layer switched to 2 outputs, as we have only two classes. The models are trained from zero, but we left the option in the notebook to use a pretrained model for those who wish to experiment.

## Hyperparameters for both models
The hyper parameters for both models are the same:
| Parameter             | Value
| :-------------------- | :----------
| `batch_size`          | 64
| `num_epochs`          | 25
| `optimizer`           | SGD
| `lr`                  | 0.001
| `momentum`            | 0.9
| `validation-split`    | 0.2

## Classifier 1
Named 'model_orig' in the notebook. The classifier will be trained on the original (hence the name) training set of 300 ice-cream images and 300 waffle images.
## Classifier 2
Named 'model_gan' in the notebook. The classifier will be trained on the enriched training set of 300(orig) + 1000(artificial) samples for each class.
### Creating the artificial data:
We used the automatic parameters for the GAN training with the exception of training with 'kimg=1000' - as this parameter directly affects how long the model would train and the quality of the generated images.
The GAN training initialization is relatively straight-forward in our case, and is fully described in this code snippet:
```.bash
# convert data to standard format
python dataset_tool.py --source=./saved_images_dir --dest=./gan_in

# train model for kimg=1000 (kimg parameter effectively controls training time and quality)
python train.py --outdir=./trained_gan --data=./gan_in --kimg=1000

# generate 1000 random images from the trained latent space
python generate.py --outdir=generated_out --network=./trained_gan --seed=1-1000
```
notes:

./saved_images_dir - should contain the images saved after crop and normalization by pytorch (set need_save=True in the notebook and specify path).

./generated_out - will contain 1000 new artificially generated images.

### Merging the data
After generating seperate images for both classes you can merge them with the original training data to create the modified dataset for the second classifier.
Thus, as a result, the second classifier shall contain 2600 training images as opposed to 600 of the first classifier.

### Some artificially generated examples (cherry picked):
Ice-cream:

![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/ic/seed0003.png)
![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/ic/seed0224.png)
![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/ic/seed0479.png)
![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/ic/seed0956.png)

Waffles:

![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/w/seed0011.png)
![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/w/seed0020.png)
![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/w/seed0082.png)
![alt text](https://github.com/fallenshock/DL_Course_Project/blob/main/GAN_results/w/seed0268.png)

## Project flow
The whole project was done in colab using the supplied notebook (training_classifiers.ipynb) - With the exception of training the GAN, which was executed on a remote machine because of the time constraints imposed by Google Colab, the code for running the GAN (after cloning the StyleGAN2-ADA repository) is given in the 'Creating the artificial data' section and also inside a markdown cell in the notebook at the appropriate section.


# Results
We've conducted several experiments with different number of epochs and number of fake images generated by the GAN:

|exp No.| Num Epochs |fake data (for each class)| model_orig Test Accuracy | model_gan Test Accuracy 
| :-----| :--------- | :----------------------- | :----------------------- | :------------------------
| `1`   | `15`       | `1000`                   | 83.67%                   | 85.71%
| `2`   | `15`       | `2000`                   | 81.63%                   | 87.76%
| `3`   | `25`       | `2000`                   | 82.65%                   | 82.65%

Reminder : `model_orig` is the first classifier, that is trained only on the original data. `model_gan` is trained on both original and the GAN generated data.
* note:* there's a slight variance in test accuracy for model_orig in experiments 1 & 3 - we expected the same accuracy but the randomized validation set might be the cause for the difference.

## graphs
exp No. 1 val_accuracy vs epoch
<figure>
  <img src="https://github.com/fallenshock/DL_Course_Project/blob/main/graphs/graph_83_85.png"  />
</figure>

exp No. 2 val_accuracy vs epoch
<figure>
  <img src="https://github.com/fallenshock/DL_Course_Project/blob/main/graphs/graph_81_87.png"  />
</figure>

exp No. 3 val_accuracy vs epoch
<figure>
  <img src="https://github.com/fallenshock/DL_Course_Project/blob/main/graphs/graph_82_82.png"  />
</figure>

## Comparing results with classic data augmentations
We also conducted an experiment where we train the model on the original train set with classic data augmentations: Color jitter and random horizontal flip:
|exp No.| Num Epochs |augmentations                         | model_orig Test Accuracy 
| :-----| :--------- | :----------------------------------- | :----------------------- 
| `4`   | `25`       | `ColorJitter`,`RandomHorizontalFlip` | 86.73%                   

## Discussion
We can see that a better test accuracy is usually achieved when using StyleGAN2-Ada to create fake training images.
We also noticed that the best results were for `model_gan` with 2000 fake images for each class and 15 training epochs. When we increase the number of epochs, the model starts to lose it's edge over the original model. We assume that this is because `model_gan` starts to overfit on the fake data, and gets a lower score on the test set (as seen in experiment 3).
In addition, we see that more fake data yields better results on the test set. Although we must be careful not to add to much, because the risk of overfitting on the fake data becomes more significant.
Also, using classic data augmentation achieves a similar improvement on the test accuracy (86.73%) - Only slightly worse than the results with StyleGAN2-ADA.

# Conclusion

In this project, we we've shown a proof of concept that using StyleGAN2-ADA to increase the size of a small training set for an image classification task improved the final test accuracy by 2-5% on average. Of course, this was tested only on one model (VGG16) with mostly fixed hyperparameters , and on one dataset (waffles-or-icecream by Sapal6).
The results were compared with classic data augmentations (color jitter and random H-flip) and it was shown that the GAN method yielded slighty better results.

## Future work

This project can be expanded to experiment with:
- Different Classifier Architectures & Transfer Learning (Code ready in notebook).
- Different Classifier Hyperparameters such as lr, scheduler, optimizer, data augmentations etc.
- Train StyleGAN2-ADA for longer ('kimg' parameter) - Read documentation in their git.
- Experiment with a different dataset.
- test with a larger generated dataset for each class


