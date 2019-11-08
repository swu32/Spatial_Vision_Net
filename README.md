# Spatial-vision Net

## Introduction

This code package contains the code for training Spatial Vision Net and its variations trained on CIFAR10 Dataset. 

## Run

To train a Spatial-Vison_Net on Imagenet dataset with image size of 128 x 128, and batchsize of 64, run: 

```shell
python3 training.py -a SV_Net_I -d ImageNet128 -b 64
```



## File Description: 

main.py 

Main function call to train the configured network. 

model.py

containes all available types of spatial-vision models, in addition to vanilla resnet. 



## Spatial-vision Net

Spatial Vision Net Integrates a front end of a computational spatial vision model with a backend of ResNet 18. 

![Spatial-vision Net Architecture](images/Spatial-vision_Net_architecture.png)Spatial-vision Net essentially contains 3 components, decomposition, normalization and nonlinearity. 

### Decomposition: 

First, the colored image is convered into luminance gray images. Then, the gray scale image is convolved with each filters from our Log-Gabor filter bank with 12 frequencies, 8 orientations, and 2 phases. In total, 192 filters are used to create 192 times over-complete representation $$A$$ of the original image.  

![Spatial-vision Net Architecture](images/Decomposition.png)



### Normalization:  

![Spatial-vision Net Architecture](images/Normalization.png)

$$A$$ is a tensor with activities of the 192 filters applied as a result of convolution. A is passed onto one level of nonlinearity. 

The purpose of Normalization is to obtain a weighted average of each pixel with respected to its neighbors in two spatial dimensions, frequency and orientation. The weight matrix is defined as a 4 dimensional gaussian. As a result of the convolution between $A^p$ and $G$, each pixel of $B$ can be understood as a smoothified version of its corresponding pixel in $A^p$, with its value depending most strongly on the close neighbor of its corresponding pixel in $A^p$, and less strongly on far neighbor of its corresponding pixel in $A^p$.

### Nonlinearity:

Nonlinearity is a process that models the competition between neurons. For a signal to stand out, The ratio between elements of $A$ and elements of $B$ can be understood as computing how an element in a high dimensional tensor stands out compared to its weighted averaged neighbors. If this element is significantly bigger than its weighted averaged version, the ratio is going to be bigger, compared to another element which is less than its weighted averaged version. Constant $c$ is used in the denumerator to protect it against  numerical instability

$B$ is a smoothified version of $A$. 



![Spatial-vision Net Architecture](images/Nonlinearity.png)

