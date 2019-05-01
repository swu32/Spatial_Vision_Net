"""This code is adapted from the resnet.py file in torchvision.model"""
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.io
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo







def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class V1_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, n_img_per_batch = 4, n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224):
        self.inplanes = 64
        super(V1_ResNet, self).__init__()
        self.n_img_per_batch = n_img_per_batch
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.imsize = imsize

        self.v1 = V1_Imagenet_net(n_img_per_batch = self.n_img_per_batch, n_freq  = self.n_freq, n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize)
        self.conv1 = nn.Conv2d(n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.v1(x) # (n_batch,n_feature, featureoutput, featureoutput)
        x = self.conv1(x) # (n_batch,64, convolution_output, convolution_output)

        x = self.bn1(x) 
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # ?
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class mean_padding(torch.nn.Module): # checked
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
        mean_batch = torch.mean(torch.mean(x,3),2).view([self.n_img_per_batch,1,1,1])
        print(type(self.padding),'self.padding')
        print(type(mean_batch),'mean_batch')
        mean_pad = torch.mul(self.padding,mean_batch)
        mean_pad[:,:,self.pad_l:-self.pad_l,self.pad_l:-self.pad_l] = x
        
        return mean_pad


class log_Gabor_convolution(torch.nn.Module): # checked
    def __init__(self, sz, n_img_per_batch):
        """
        include declarations that can be prespeified. 
        """
        super(log_Gabor_convolution, self).__init__()
        pad_l = int(sz/2) # pad half of the image size. 
        self.mean_padding = mean_padding(pad_l, n_img_per_batch,sz)
        self.combined_filters = self.load_filter_bank()
        self.n_img_per_batch = n_img_per_batch
        print(self.n_img_per_batch)
        
    def load_filter_bank(self,path ='./spatial_filters_224_by_224.mat'):
        '''Assumes a certain structure of filter file, returns a filter bank'''
        mat = scipy.io.loadmat(path)
        spatial_filters_imag = mat['spatial_filters_imag'] 
        spatial_filters_real = mat['spatial_filters_real']
        n_freq = spatial_filters_imag.shape[0]
        n_orient = spatial_filters_imag.shape[1]
        sz = spatial_filters_imag.shape[2] # Image size
        n_filters = n_freq*n_orient*2 # multiply by phase
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
        convolve x with shape [n_imag_per_batch, 1, 2*imgsize, 2*imgsize]
        with hard coded filters with shape [2*n_freq*n_orient, 1, imgsize, img_size]
        returns a with shape [n_imag_per_batch,2*n_freq*n_orient,imgsize+1, img_size+1]
        """
        print(x.shape[0],print(self.n_img_per_batch))
        assert self.n_img_per_batch == x.shape[0], "batch size needs to match the zeroth diension of x"
        mean_padded_images = self.mean_padding(x)
        a = F.conv2d(mean_padded_images, self.combined_filters)   
        print(a.shape)
        return a
    

# example module, skip for now. 
class Normalization(nn.Module):
    """Calculated Normalized Layer, essential the coefficients b_i,
    apply convolution with 4d gaussian with a^p, where a is the result of convolution with customized filter."""
    def __init__(self,p = 2,sz = 224,n_img_per_batch = 4, l_x = 32,l_y = 32,l_f = 3,l_o = 3,padxy_l = 16,padxy_r = 15,padxy_t = 16,padxy_b = 15,pad_fo = 1,w_x = 1,
    w_y = 1,w_f = 1,w_o = 1):
        """
        p: power to multiply
        sz: image size (assume square images)
        n_img_per_batch: number of images per batch
        l_x/l_y/l_f/l_o: gaussian convolution spaceing of x/y/f/o between -1 and 1
        padxy_l/padxy_r/padxy_t/padxy_b: xy pad left/right/top/bottom
        pad_fo: frequency/orientation pad, assume left = right = top = bottom
        w_x/w_y/w_f/w_o: standard deviation of gaussian in x/y/f/o
        """
        super(Normalization, self).__init__()
        self.l_x = l_x
        self.l_y = l_y
        self.l_f = l_f
        self.l_o = l_o
        self.sz = sz
    
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
        self.n_img_per_batch = n_img_per_batch
        self.n_feature = self.n_phase*self.n_orient*self.n_freq
        self.convx_sz = self.sz+self.padxy_t+self.padxy_b-self.l_y + 1
        self.convy_sz = self.sz+self.padxy_l+self.padxy_r-self.l_x + 1
        

    def forward(self, x):
        """shape of x: ([n_imag_per_batch, 2*n_freq*n_orientation, imsize, imsize])
            Normalize x^p, correspond to equation (1)
            
            returns B with shape torch.Size([n_img_per_batch, n_freq, n_orient, n_phase, sz_after_convolution, sz_after_convolution])"""
        assert x.shape[2] == self.sz
        assert x.shape[3] == self.sz
        assert x.shape[0] == self.n_img_per_batch
        assert x.shape[1] == self.n_feature
        
        x_p = x**self.p
        n_img_per_batch = x.shape[0]
        n_feature = x.shape[1]
        imsize = x.shape[2] # assumes square image
        
        ## convolution in x
        permuted = self.padxy(x_p).permute([1,0,2,3])
        permute_concat = permuted.view([self.n_img_per_batch*self.n_feature,1,self.sz+self.padxy_t+self.padxy_b,self.sz+self.padxy_l+self.padxy_r])
        conv_x = F.conv2d(permute_concat, self.Gauss_x) 
        conv_y = F.conv2d(conv_x, self.Gauss_y)
        conv_y_ = conv_y.view([n_feature,n_img_per_batch,self.convx_sz,self.convy_sz])
        conv_y__ = conv_y_.view([self.n_freq,self.n_orient,self.n_phase,n_img_per_batch,1,self.convx_sz,self.convy_sz])
        conv_y___ = conv_y__.permute([5,6,2,3,4,0,1])
        conv_y____ = conv_y___.view([-1,1,self.n_freq,self.n_orient]) # [convx_sz*convy_sz*n_phase*n_img_per_batch,1,self.n_freq, self.n_orientation]
        ## convolution in f
        conv_f = F.conv2d(self.padfo(conv_y____), self.Gauss_f)
        conv_o = F.conv2d(conv_f, self.Gauss_o)
        B = conv_o.view([self.convx_sz,self.convy_sz,self.n_phase,self.n_img_per_batch,self.n_freq,self.n_orient])
        B_ = B.permute([3,4,5,2,0,1])
    
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

    def forward(self, A, B):
        """The shape of a and b needs to be the same
            [n_image_per_batch, n_freq,_n_orientation,n_phase, width_after_convolution, height_after_convolution]"""
        assert A.shape ==B.shape, "The shape of a and b needs to be the same"
        Addition = torch.add(torch.tensor(self.C**self.p),B) # Addition is ok, no nans 
        r = torch.div(torch.pow(A,(self.p+self.q)),torch.add(torch.tensor(self.C**self.p),B))        
        return r

def v1resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = V1_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

# check the log gabor convolution works or not
class V1_Imagenet_net(nn.Module):
    def __init__(self,n_img_per_batch = 4, n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224):
        super(V1_Imagenet_net, self).__init__()
        self.n_img_per_batch = n_img_per_batch
        self.imsize = imsize
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.conv_after_x = self.imsize*2 - self.imsize + 1
        self.conv_after_y = self.conv_after_x # assume square images
        self.logabor = log_Gabor_convolution(imsize,n_img_per_batch)
        self.sz_after_filtering = self.imsize*2 - self.imsize + 1
        self.normalization =  Normalization(sz = self.sz_after_filtering,n_img_per_batch = n_img_per_batch)
        self.nonlinearity = Nonlinearity()

    def forward(self, images):
        # [4,3,32,32]
        a_ = self.logabor(images)
        B_normalization = self.normalization(a_) # the same till here
        A = a_.view([self.n_img_per_batch,self.n_freq,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y])
        R = self.nonlinearity(A,B_normalization)
        R_reshape = R.view([self.n_img_per_batch,-1,self.conv_after_x,self.conv_after_y])
        # [n_img_per_batch, 192, 225, 225] 

        return R_reshape

