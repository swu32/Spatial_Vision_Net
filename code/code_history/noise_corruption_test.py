import imagenet_c
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision 
import torch
"""Test file """
'''noise function that can be used:  'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur','glass_blur','fog','speckle_noise', 'gaussian_blur''''
transform1 = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]) # transforms sequentially, mean and variance prespecified 
trainset = torchvision.datasets.CIFAR10(root = './data',train = True, download = False, transform = transform1 )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
# print(images.shape)
this_image =images[3,0,:,:]
print('this image',this_image)
plt.subplot(121)
plt.imshow(this_image,cmap = 'gray')
K = imagenet_c
M = K.corrupt(this_image, severity=5, corruption_name="saturate")
print('the shape of M',M.shape)
plt.subplot(122)

plt.imshow(M,cmap = 'gray')
plt.show()




# noise function that cannot be used: " brightness, motion_blur, zoom_blur, snow, frost, contrast, elastic_transform, pixelate, jpeg_compression, spatter, saturate"