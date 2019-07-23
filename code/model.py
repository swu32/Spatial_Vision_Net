"""This code is adapted from the resnet.py file in torchvision.model"""
'adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py'
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



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}





class Spatial_Vision_Net(nn.Module):
    '''Main Class for Spatial Vision Net'''

    def __init__(self, block, layers, low_freq = False, num_classes=1000, n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224):
        self.inplanes = 64
        super(Spatial_Vision_Net, self).__init__()
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.imsize = imsize
        if low_freq: # only employs the lower half of the spatial frequency filters, 
            self.v1 = V1_Low_Frequency_net(n_freq  = int(self.n_freq/2), n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize)
            self.conv1 = nn.Conv2d(int(self.n_freq/2)*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3,bias=False)
        else:
            self.v1 = V1_Imagenet_net(n_freq  = self.n_freq, n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize)
            self.conv1 = nn.Conv2d(n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3,bias=False)

        # specify parameters for the rest of the network
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 *4* block.expansion, num_classes)


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
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride= stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Spatial_Vision_Net_II(nn.Module):
    '''An upgrated version of spatial vision net, where boundary effect is alievated, positive and negative arctivities are 
        separated, and backend is changed to a simpler architecture. '''

    def __init__(self, block, num_blocks, num_classes=1000, n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224,low_freq = False):
        self.inplanes = 64
        super(Spatial_Vision_Net_II, self).__init__()
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.imsize = imsize

        if low_freq: # only employs the lower half of the spatial frequency filters, 
            self.v1 = SV_net(n_freq  = int(self.n_freq/2), n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize, low_freq = True)
            self.conv1 = nn.Conv2d(2*int(self.n_freq/2)*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)
            self.n_freq = int(self.n_freq/2)
        else:
            self.v1 = SV_net(n_freq  = self.n_freq, n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize)
            self.conv1 = nn.Conv2d(2*n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)

        self.relu = nn.ReLU(inplace=True)
        # modify this part: 
        self.in_planes = 2*self.n_freq*n_orient*n_phase
        self.bn1 = nn.BatchNorm2d(2*self.n_freq*n_orient*n_phase)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # 2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # 2
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # 2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # 2
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.v1(x) # (n_batch,n_feature, featureoutput, featureoutput)
        x = self.bn1(x) # torch.Size([5, 384, 33, 33])
        x = self.layer1(x) # ([5, 64, 33, 33])
        x = self.layer2(x) # [5, 128, 17, 17]
        x = self.layer3(x) # [5, 256, 9, 9]
        x = self.layer4(x) # [5, 512, 5, 5]
        x = F.avg_pool2d(x, 4) # ([5, 512, 1, 1])
        x = x.view(x.size(0), -1) # torch.Size([5, 512])
        x = self.linear(x)
        return x


class Spatial_Vision_Net_III(nn.Module):
    '''An extremely simplified version of SV_net; used for MNIST'''

    def __init__(self, num_classes=10, n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 32):
        super(Spatial_Vision_Net_III, self).__init__()
        self.inplanes = 64
        self.n_phase = n_phase
        self.imsize = imsize
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.v1 = SV_net(n_freq  = int(self.n_freq/2), n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize, low_freq = True)
        self.conv1 = nn.Conv2d(n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p = 0.5)
        self.conv2 = nn.Conv2d(64, 16, 2)
        self.fc1 = nn.Linear(16 * 3 * 3, num_classes)


    def forward(self, x):
        x = self.v1(x) # (n_batch,n_feature, featureoutput, featureoutput)
        x = self.pool(F.relu(self.conv1(self.dropout(x))))
        x = self.pool(F.relu(self.conv2(self.dropout(x))))
        x = x.view(-1, 16 * 3 * 3)
        x = self.fc1(self.dropout(x))
        return x





class mean_padding(torch.nn.Module): # checked
    def __init__(self,pad_l,sz):
        """
        padd each image with their mean, with their specified parameters,
        
        pad_l: padding length, this function assumes a squared equal padding on all sides of image 
        sz: size of image, this function assumes a square image. 
        """
        super(mean_padding, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.padding = torch.ones([1,1,sz+pad_l*2,sz+pad_l*2]).to(device)
        self.pad_l = pad_l

    def forward(self, x):
        """
        """
        n_img_per_batch = x.shape[0] #this has to be true
        print('n image per batch is ', n_img_per_batch)
        assert x.shape[2] == x.shape[3],'The image to be mean padded should be square sized'
        mean_batch = torch.mean(torch.mean(x,3),2).view([n_img_per_batch,1,1,1])
        mean_batch = torch.mul(self.padding,mean_batch)
        mean_batch[:,:,self.pad_l:-self.pad_l,self.pad_l:-self.pad_l] = x
        
        return mean_batch


class log_Gabor_convolution(torch.nn.Module): # checked
    def __init__(self, sz = 32, low_freq = False):
        """
        low_freq: only use half of the filters starting with the lowest frequency
        """
        super(log_Gabor_convolution, self).__init__()
        pad_l = int(sz/2) # pad half of the image size. 
        self.low_freq = low_freq
        self.mean_padding = mean_padding(pad_l,sz)
        self.combined_filters = self.load_filter_bank(low_freq = self.low_freq, sz = sz)
        # self.batchsize = batchsize
        

    def load_filter_bank(self,low_freq = False, sz = 32):

        '''Assumes a certain structure of filter file, returns a filter bank'''
        if sz == 32: # load corresponding filter bank
            path ='./spatial_filters.mat'
        else:
            path = './spatial_filters_224_by_224.mat'

        mat = scipy.io.loadmat(path)
        spatial_filters_imag = mat['spatial_filters_imag'] 
        spatial_filters_real = mat['spatial_filters_real']
        n_freq = spatial_filters_imag.shape[0]
        n_orient = spatial_filters_imag.shape[1]
        sz = spatial_filters_imag.shape[2] # Image size
        n_filters = n_freq*n_orient*2 # multiply by phase
        # combine real and imagary filters
        if not low_freq:
            banksize = n_filters
            start = 0
        else: 
            banksize = int(n_filters/2)
            start = int(n_freq/2)

        # modified code: 
        filter_banks = np.zeros([banksize,1,sz,sz]) # 1 is left for the gray value channel
        s = 0

        for f in range(start,n_freq):
            for o in range(0, n_orient):
                filter_banks[s,0,:,:] = spatial_filters_real[f,o] - np.mean(spatial_filters_real[f,o])
                s = s + 1
                filter_banks[s,0,:,:] = spatial_filters_imag[f,o] - np.mean(spatial_filters_imag[f,o])
                s = s + 1

        filter_banks = torch.FloatTensor(filter_banks)
        # filter_banks = filter_banks.type(torch.FloatTensor)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        filter_banks = filter_banks.to(device)
        return filter_banks

    def forward(self, x):
        """
        convolve x with shape [n_imag_per_batch, 1, 2*imgsize, 2*imgsize]
        with hard coded filters with shape [2*n_freq*n_orient, 1, imgsize, img_size]
        returns a with shape [n_imag_per_batch,2*n_freq*n_orient,imgsize+1, img_size+1]
        """
        # assert self.batchsize == x.shape[0], "batch size needs to match the zeroth dimension of x"
        x = self.mean_padding(x)
        x = F.conv2d(x, self.combined_filters)
        return x
    

# example module, skip for now. 
class Normalization(nn.Module):
    """Calculated Normalized Layer, essential the coefficients b_i,
    apply convolution with 4d gaussian with a^p, where a is the result of convolution with customized filter."""
    def __init__(self, n_freq = 12, p = 2,sz = 224, l_x = 32,l_y = 32,l_f = 3,l_o = 3,padxy_l = 16,padxy_r = 15,padxy_t = 16,padxy_b = 15,pad_fo = 1,w_x = 1,
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        Gauss_x = 1/(np.sqrt(2.0*np.pi)*w_x)*np.exp(-(x**2)/(2*w_x**2))
        self.Gauss_x = torch.from_numpy(Gauss_x.reshape([1,1,1,l_x])).float().to(device)
        Gauss_y = 1/(np.sqrt(2.0*np.pi)*w_y)*np.exp(-(y**2)/(2*w_y**2))
        self.Gauss_y = torch.from_numpy(Gauss_y.reshape([1,1,l_y,1])).float().to(device)
        Gauss_o = 1/(np.sqrt(2.0*np.pi)*w_o)*np.exp(-(o**2)/(2*w_o**2))
        self.Gauss_o = torch.tensor(Gauss_o.reshape([1,1,1,l_o])).type(torch.FloatTensor).to(device)
        Gauss_f = 1/(np.sqrt(2.0*np.pi)*w_f)*np.exp(-(f**2)/(2*w_f**2))
        self.Gauss_f = torch.tensor(Gauss_f.reshape([1,1,l_f,1])).type(torch.FloatTensor).to(device)
        self.padxy = nn.ReflectionPad2d((padxy_l,padxy_r,padxy_t,padxy_b))
        self.padfo = nn.ConstantPad2d(pad_fo, 0) # frequency/orientation 2d pad
        self.padxy_l = padxy_l
        self.padxy_r = padxy_r
        self.padxy_t = padxy_t
        self.padxy_b = padxy_b
        self.n_freq = n_freq
        self.n_orient = 8
        self.n_phase = 2
        self.p = p
        self.sz = sz
        self.n_feature = self.n_phase*self.n_orient*self.n_freq
        self.convx_sz = self.sz+self.padxy_t+self.padxy_b-self.l_y + 1
        self.convy_sz = self.sz+self.padxy_l+self.padxy_r-self.l_x + 1
        

    def forward(self, x):
        """shape of x: ([n_imag_per_batch, 2*n_freq*n_orientation, imsize, imsize])
            Normalize x^p, correspond to equation (1)
            
            returns B with shape torch.Size([n_img_per_batch, n_freq, n_orient, n_phase, sz_after_convolution, sz_after_convolution])"""
        #assert x.shape[2] == self.sz
        #assert x.shape[3] == self.sz
        #assert x.shape[0] == self.n_img_per_batch
        #assert x.shape[1] == self.n_feature
        batchsize = x.shape[0]
        #imsize = x.shape[2] # assumes square image
        x_p = x**self.p
        ## convolution in x
        B = self.padxy(x_p).permute([1,0,2,3]).contiguous()
        B = B.view([-1,1,self.sz+self.padxy_t+self.padxy_b,self.sz+self.padxy_l+self.padxy_r])
        B = F.conv2d(B, self.Gauss_x) 
        B = F.conv2d(B, self.Gauss_y).contiguous()
        B = B.view([self.n_feature,batchsize,self.convx_sz,self.convy_sz])
        B = B.view([self.n_freq,self.n_orient,self.n_phase,batchsize,1,self.convx_sz,self.convy_sz])
        B = B.permute([5,6,2,3,4,0,1]).contiguous()
        B = B.view([-1,1,self.n_freq,self.n_orient]) # [convx_sz*convy_sz*n_phase*n_img_per_batch,1,self.n_freq, self.n_orientation]
        ## convolution in f
        B = F.conv2d(self.padfo(B), self.Gauss_f)
        B = F.conv2d(B, self.Gauss_o).contiguous()
        B = B.view([self.convx_sz,self.convy_sz,self.n_phase,batchsize,self.n_freq,self.n_orient])
        B = B.permute([3,4,5,2,0,1])
    
        return B

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
        #assert A.shape ==B.shape, "The shape of a and b needs to be the same"
        #Addition = torch.add(torch.tensor(self.C**self.p),B) # Addition is ok, no nans 
        r = torch.div(torch.pow(A,(self.p+self.q)),torch.add(torch.tensor(self.C**self.p),B))        
        return r

def v1resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    model = Spatial_Vision_Net(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.to(device)
    return model

def low_freq_resnet18(pretrained=False, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Spatial_Vision_Net(BasicBlock, [2, 2, 2, 2], low_freq = True, **kwargs)
    model.to(device)
    return model

def simple_net(pretrained=False, **kwargs):
    # a net with positive and negative signal separation and a simple backend. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Spatial_Vision_Net_II(BasicBlock, [2, 2, 2, 2],**kwargs)
    model.to(device)
    return model

def low_freq_simple_net(pretrained=False, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Spatial_Vision_Net_II(BasicBlock, [2, 2, 2, 2], low_freq = True, **kwargs)
    model.to(device)
    return model


def simple_net_III(pretrained=False, **kwargs):
    # designed for MNIST experiments
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Spatial_Vision_Net_III(**kwargs)
    model.to(device)
    return model

# check the log gabor convolution works or not
class V1_Imagenet_net(nn.Module):
    def __init__(self,n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224):
        super(V1_Imagenet_net, self).__init__()
        self.imsize = imsize
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.conv_after_x = self.imsize*2 - self.imsize + 1
        self.conv_after_y = self.conv_after_x # assume square images
        self.logabor = log_Gabor_convolution(sz = imsize)
        self.sz_after_filtering = self.imsize*2 - self.imsize + 1
        self.normalization =  Normalization(sz = self.sz_after_filtering)
        self.nonlinearity = Nonlinearity()

    def forward(self, images):
        # [4,3,32,32]
        batchsize = images.shape[0]
        a_ = self.logabor(images).contiguous()
        B_normalization = self.normalization(a_) # the same till here
        A = a_.view([batchsize,self.n_freq,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y])
        R = self.nonlinearity(A,B_normalization).contiguous()
        R = R.view([batchsize,-1,self.conv_after_x,self.conv_after_y])
        # [n_img_per_batch, 192, 225, 225] 

        return R

class SV_net(nn.Module):
    '''implement the spatial vision subcomponent of the module
    separate positive and negative subpart of the response before convolution'''
    def __init__(self,n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224, low_freq = False):
        super(SV_net, self).__init__()
        self.imsize = imsize
        self.n_freq = n_freq
        #print('self.n_freq',self.n_freq)
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.conv_after_x = self.imsize*2 - self.imsize + 1
        self.conv_after_y = self.conv_after_x # assume square images
        if low_freq:    
            self.logabor = log_Gabor_convolution(sz = imsize, low_freq = True)
        else:
            self.logabor = log_Gabor_convolution(sz = imsize)

        self.sz_after_filtering = self.imsize*2 - self.imsize + 1
        self.normalization =  Normalization(n_freq = self.n_freq, sz = self.sz_after_filtering)
        self.nonlinearity = Nonlinearity()
    def sign_segragation(self,this_filter_o):
        # input: some filters output
        # output, positive and negative filter activities, to be processed separately
        sign = this_filter_o>0
        if torch.cuda.is_available():
            sign = sign.type(torch.cuda.FloatTensor)
        else:
            sign = sign.type(torch.FloatTensor)

        positive = this_filter_o*sign
        negative = -1*this_filter_o*(1-sign)
        return positive, negative # returns only positive values for competition/gain control

    def forward(self, images):
        # separate positive and negative part of a_ for normalization
        # [4,3,32,32]
        batchsize = images.shape[0]

        a_ = self.logabor(images).contiguous()
        #print("shape of a_", a_.shape)
        a_pos, a_neg =self.sign_segragation(a_)
        #print('shape of signed ',a_pos.shape)
        A_pos = a_pos.view([batchsize,self.n_freq,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y])
        A_neg = a_neg.view([batchsize,self.n_freq,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y])

        B_normalization_pos = self.normalization(a_pos) # the same till here
        B_normalization_neg = self.normalization(a_neg) # the same till here

        R_pos = self.nonlinearity(A_pos,B_normalization_pos).contiguous()
        R_neg = self.nonlinearity(A_neg,B_normalization_neg).contiguous()

        R_pos = R_pos.view([batchsize,-1,self.conv_after_x,self.conv_after_y])
        R_neg = R_neg.view([batchsize,-1,self.conv_after_x,self.conv_after_y])
        R = torch.cat((R_pos,R_neg),1)
        # [n_img_per_batch, 192*2, 32, 32] 

        return R


# Module only using low frequency filters
class V1_Low_Frequency_net(nn.Module):
    # only use half of the frequency for classification
    def __init__(self,n_freq  = 6, n_orient = 8, n_phase = 2, imsize = 224):
        super(V1_Low_Frequency_net,self).__init__()
        #print('it is ok till here')
        self.imsize = imsize
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.conv_after_x = self.imsize*2 - self.imsize + 1
        self.conv_after_y = self.conv_after_x # assume square images
        self.logabor = log_Gabor_convolution(sz = imsize, low_freq = True)
        self.sz_after_filtering = self.imsize*2 - self.imsize + 1
        self.normalization =  Normalization(n_freq = self.n_freq, sz = self.sz_after_filtering)
        self.nonlinearity = Nonlinearity()

    def forward(self, images):
        # [4,3,32,32]
        batchsize = images.shape[0]
        a_ = self.logabor(images).contiguous()
        #print('shape of a ', a_.shape) # [4, 96, 33, 33 ]
        B_normalization = self.normalization(a_) # the same till here
        A = a_.view([batchsize,self.n_freq,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y])
        R = self.nonlinearity(A,B_normalization).contiguous()
        R = R.view([batchsize,-1,self.conv_after_x,self.conv_after_y])
        # [n_img_per_batch, 96, 225, 225] 
        return R



"""ResNet related functions"""
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

