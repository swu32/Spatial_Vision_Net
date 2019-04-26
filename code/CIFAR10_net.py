import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class mean_padding(torch.nn.Module): # all parameters do not require gradients
    def __init__(self,pad_l,n_img_per_batch,sz):
        """
        padd each image with their mean, with their specified parameters,
        
        pad_l: padding length, this function assumes a squared equal padding on all sides of image 
        sz: size of image, this function assumes a square image. 
        """
        super(mean_padding, self).__init__()
        self.padding = torch.ones([1,1,sz+pad_l*2,sz+pad_l*2])
        self.pad_l = pad_l
        self.n_img_per_batch = n_img_per_batch

    def forward(self, x):
        """
        """
        assert x.shape[2] == x.shape[3],'The image to be mean padded should be square sized'
        n_img_per_batch = x.shape[0]
        mean_batch = torch.mean(torch.mean(x,3),2).reshape([self.n_img_per_batch,1,1,1])
        mean_pad = torch.mul(self.padding,mean_batch)
        mean_pad[:,:,self.pad_l:-self.pad_l,self.pad_l:-self.pad_l] = x
        return mean_pad
    
class log_Gabor_convolution(torch.nn.Module):
    def __init__(self, sz, n_img_per_batch):
        """
        include declarations that can be prespeified. 
        """
        super(log_Gabor_convolution, self).__init__()
        pad_l = int(sz/2) # pad half of the image size. 
        self.mean_padding = mean_padding(pad_l, n_img_per_batch,sz)
        self.combined_filters = self.load_filter_bank()
        self.n_img_per_batch = n_img_per_batch


    def load_filter_bank(self,path ='./spatial_filters.mat'):
        '''Assumes a certain structure of filter file, returns a filter bank'''
        mat = scipy.io.loadmat(path)
        spatial_filters_imag = mat['spatial_filters_imag'] 
        spatial_filters_real = mat['spatial_filters_real']
        n_freq = spatial_filters_imag.shape[0]
        n_orient = spatial_filters_imag.shape[1]
        sz = spatial_filters_imag.shape[2] # Image size
        n_filters = n_freq*n_orient*2
        # combine real and imagary filters
        filter_banks = np.zeros([n_filters,1,sz,sz]) # 1 is left for the gray value channel

        s = 0
        for f in range(0,n_freq):
            for o in range(0, n_orient):
                filter_banks[s,0,:,:] = spatial_filters_real[f,o] - np.mean(spatial_filters_real[f,o])
                s = s + 1
                filter_banks[s,0,:,:] = spatial_filters_imag[f,o] - np.mean(spatial_filters_imag[f,o])
                s = s + 1

        filter_banks = torch.tensor(filter_banks)
        filter_banks = filter_banks.type(torch.FloatTensor)

        return filter_banks

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        assert self.n_img_per_batch == x.shape[0], "batch size needs to match the zeroth diension of x"

        mean_padded_images = self.mean_padding(x)
        a = F.conv2d(mean_padded_images, self.combined_filters)   
#         print(mean_padded_images.requires_grad)
#         print(self.combined_filters.requires_grad)
#         print(a.requires_grad)
        return a
    

# example module, skip for now. 
class Normalization(nn.Module):
    """Calculated Normalized Layer, essential the coefficients b_i,
    apply convolution with 4d gaussian with a^p, where a is the result of convolution with customized filter."""
    def __init__(self,p = 2,sz = 32,l_x = 32,l_y = 32,l_f = 3,l_o = 3,padxy_l = 16,padxy_r = 15,padxy_t = 16,padxy_b = 15,pad_fo = 1,w_x = 1,
    w_y = 1,w_f = 1,w_o = 1):
        """
        p: power to multiply
        l_x/l_y/l_f/l_o: number of x/y/f/o between -1 and 1 for gaussian
        padxy_l/padxy_r/padxy_t/padxy_b: xy pad left/right/top/bottom
        pad_fo: frequency/orientation pad, assume left = right = top = bottom
        w_x/w_y/w_f/w_o: standard deviation of gaussian in x/y/f/o
        """
        super(Normalization, self).__init__()
        x = np.linspace(-1,1,l_x)
        y = np.linspace(-1,1,l_y)
        f = np.linspace(-1,1,l_f)
        o = np.linspace(-1,1,l_o)

        Gauss_x = 1/(np.sqrt(2.0*np.pi)*w_x)*np.exp(-(x**2)/(2*w_x**2))
        self.Gauss_x = torch.from_numpy(Gauss_x.reshape([1,1,1,l_x])).float()
        Gauss_y = 1/(np.sqrt(2.0*np.pi)*w_y)*np.exp(-(y**2)/(2*w_y**2))
        self.Gauss_y = torch.from_numpy(Gauss_y.reshape([1,1,l_y,1])).float()
        Gauss_o = 1/(np.sqrt(2.0*np.pi)*w_o)*np.exp(-(o**2)/(2*w_o**2))
        self.Gauss_o = torch.tensor(Gauss_o.reshape([1,1,1,l_o])).type(torch.FloatTensor)
        Gauss_f = 1/(np.sqrt(2.0*np.pi)*w_f)*np.exp(-(f**2)/(2*w_f**2))
        self.Gauss_f = torch.tensor(Gauss_f.reshape([1,1,l_f,1])).type(torch.FloatTensor)
        self.padxy = nn.ConstantPad2d((padxy_l,padxy_r,padxy_t,padxy_b), 0)
        self.padfo = nn.ConstantPad2d(pad_fo, 0) # frequency/orientation 2d pad

        self.padxy_l = padxy_l
        self.padxy_r = padxy_r
        self.padxy_t = padxy_t
        self.padxy_b = padxy_b
        self.n_freq = 12
        self.n_orient = 8
        self.n_phase = 2
        self.p = p
        self.sz = sz

    def forward(self, x):
        """shape of x: ([n_imag_per_batch, 2*n_freq*n_orientation, 33, 33])
            Normalize x^p, correspond to equation (1)"""
        x_p = x**self.p
        n_imag_per_batch = x.shape[0]
        permuted = self.padxy(x_p).permute([1,0,2,3])
        permute_concat = permuted.reshape([x.shape[0]*x.shape[1],1,x.shape[2]+self.padxy_t+self.padxy_b,x.shape[3]+self.padxy_l+self.padxy_r])
        conv_x = F.conv2d(permute_concat, self.Gauss_x)
        conv_y = F.conv2d(conv_x, self.Gauss_y)
        conv_y_ = conv_y.reshape([x.shape[1],n_imag_per_batch,33,33])
        conv_y__ = conv_y_.reshape([self.n_freq,self.n_orient,self.n_phase,n_imag_per_batch,1,33,33])
        conv_y___ = conv_y__.permute([5,6,2,3,4,0,1])
        conv_y____ = conv_y___.reshape([-1,1,self.n_freq,self.n_orient])
        conv_f = F.conv2d(self.padfo(conv_y____), self.Gauss_f)
        conv_o = F.conv2d(conv_f, self.Gauss_o)
        B = conv_o.reshape([33,33,self.n_phase,n_imag_per_batch,self.n_freq,self.n_orient])
        B_ = B.permute([3,4,5,2,0,1])
#         print(B_.requires_grad) # False
#         print(self.Gauss_x.requires_grad) # False
        return B_

# example module, skip for now. 
class Nonlinearity(nn.Module):
    """Calculated r_i"""
    def __init__(self,p = 2.0,  q = 1, C = 0.25):
        """"""
        super(Nonlinearity, self).__init__()
        self.p = p
        self.q = q
        self.C = C
        # if p+q involves a square root, then it is only effective for positive numbers

    def forward(self, A, B):
        """The shape of a and b needs to be the same
            [n_image_per_batch, n_freq,_n_orientation,n_phase, width_after_convolution, height_after_convolution]"""
        assert A.shape ==B.shape, "The shape of a and b needs to be the same"
        Addition = torch.add(torch.tensor(self.C**self.p),B) # Addition is ok, no nans 
        r = torch.div(torch.pow(A,(self.p+self.q)),torch.add(torch.tensor(self.C**self.p),B))    
#         print(r.requires_grad) # False
        return r



# check the log gabor convolution works or not

# check the log gabor convolution works or not

class CIFAR10_net_without_normalization(nn.Module):
    def __init__(self,n_imag_per_batch = 4, n_freq  = 12, n_orient = 8, n_phase = 2, conv_after_x = 33, conv_after_y = 33):
        super(CIFAR10_net_without_normalization, self).__init__()
        self.n_imag_per_batch = n_imag_per_batch
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.conv_after_x = conv_after_x
        self.conv_after_y = conv_after_y
        
        self.logabor = log_Gabor_convolution(32,4)
        self.normalize = Normalization()
        self.nonlinearity = Nonlinearity()
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(192*16*16, 2000)
        self.fc2 = nn.Linear(2000, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, images):
        # [4,3,32,32]
#         print(self.n_imag_per_batch)
        a_ = self.logabor(images)
#         B_normalization = self.normalize(a_) # the same till here
        A = a_.reshape([self.n_imag_per_batch,self.n_freq,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y])
#         R = self.nonlinearity(A,B_normalization)
        R_reshape = A.reshape([4,-1,33,33])

        R_pooled = self.pool(F.relu(R_reshape)) # [4, 192, 16, 16] also the same here
        R_pooled_reshape = R_pooled.reshape([4,192*16*16])

        fc1_ = F.relu(self.fc1(R_pooled_reshape)) # [4, 120]
        fc2_ = F.relu(self.fc2(fc1_)) # [4, 120]
        fc3_ = F.relu(self.fc3(fc2_)) # [4, 120]
        return fc3_




def main():

    net = CIFAR10_net_without_normalization()
    transform = transforms.Compose(
        [transforms.Grayscale(),transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]) # transforms sequentially, mean and variance prespecified 

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss ))
                running_loss = 0.0

#             print(outputs)
    print('Finished Training')
    




if __name__ == '__main__':
    main()
