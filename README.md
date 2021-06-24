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
In this project we use a small dataset of 2 classes (Ice-cream and Waffles) created by sapa16. Link: https://www.kaggle.com/sapal6/waffles-or-icecream.
The dataset contains 343 samples for Ice-cream and 355 samples for Waffles.

We divided the first 300 images of both classes for the train set, and the remaining ~50 for the test set.

# The Classifiers
Both classifiers used are VGG16 with the last fc layer switched to 2 outputs, as we have only two classes. The models are trained from zero, but we left the option in the notebook to use a pretrained model for those who wish to experiment.

## Hyperparameters for both models
The hyper parameters for both models are the same:
| Parameter             | Valuse
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

### Some artificially generated examples:
Ice-cream:

Waffles:
## Project flow
The whole project was done in colab using the supplied notebook (training_classifiers.ipynb) - With the exception of training the GAN, which was executed on a remote machine because of the time constraints imposed by Google Colab, the code for running the GAN (after cloning the StyleGAN2-ADA repository) is given in the 'Creating the artificial data' section and also inside a markdown cell in the notebook at the appropriate section.


# Results
#




