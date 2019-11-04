import torch
import numpy as np
import scipy.io
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


"""This code is adapted from the resnet.py file in torchvision.model"""
'adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py'


"""ResNet related functions"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes) #The C inside, number of channels[N C H W]
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    # TODO: this ResNet may be the one that is standardly used for CIFAR10, check the imagenet implementation
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64 # assume input with 64 channels.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #based on paper, this should be 7x7, 64, stride 2
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
        return nn.Sequential(*layers) # TODO: Not sure what this line is doing.

    def forward(self, x):
        # TODO: check if the sizes matches the standard ResNet structure.
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SpatialVisionNet(nn.Module):
    """Main Class for Spatial Vision Net"""

    def __init__(self, block, layers, low_freq=False, num_classes=1000, n_freq=12, n_orient=8, n_phase=2, imsize=224):
        super(SpatialVisionNet, self).__init__()
        self.inplanes = 64
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.imsize = imsize
        if low_freq: # only employs the lower half of the spatial frequency filters, 
            self.v1 = V1ImagenetNet(n_freq=int(self.n_freq/2), n_orient=self.n_orient, n_phase=self.n_phase, im_size=self.imsize, low_freq=True)
            self.conv1 = nn.Conv2d(int(self.n_freq/2)*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.v1 = V1ImagenetNet(n_freq=self.n_freq, n_orient=self.n_orient, n_phase=self.n_phase, im_size=self.imsize)
            self.conv1 = nn.Conv2d(n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.v1(x) # (n_batch,n_feature, featureoutput, featureoutput)
        x = self.conv1(x) # (n_batch,64,convolution_output, convolution_output)
        x = self.bn1(x) # batch normalization
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.layer1(x) # ?
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # TODO: check the dimension of layers here, it seems to be different from ResNet.
        return x


class SpatialVisionNetII(nn.Module):
    '''An upgrated version of spatial vision net, where boundary effect is alievated, positive and negative arctivities are 
        separated, and backend is changed to a simpler architecture. '''

    def __init__(self, block, num_blocks, num_classes=1000, n_freq=12, n_orient=8, n_phase=2, imsize=128, low_freq=False):
        self.inplanes = 64
        super(SpatialVisionNetII, self).__init__()
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.imsize = imsize

        if low_freq: # only employs the lower half of the spatial frequency filters, 
            self.v1 = SvNet(n_freq=int(self.n_freq/2), n_orient=self.n_orient, n_phase=self.n_phase, imsize =self.imsize, low_freq=True)
            self.conv1 = nn.Conv2d(2*int(self.n_freq/2)*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)
            self.n_freq = int(self.n_freq/2)
        else:
            self.v1 = SvNet(n_freq=self.n_freq, n_orient=self.n_orient, n_phase=self.n_phase, imsize=self.imsize)
            self.conv1 = nn.Conv2d(2*n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)

        self.relu = nn.ReLU(inplace=True)
        # modify this part: 
        self.in_planes = 2*self.n_freq*n_orient*n_phase
        self.bn1 = nn.BatchNorm2d(2*self.n_freq*n_orient*n_phase)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)# 2
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)# 2
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
        x = self.v1(x) # (n_batch,n_feature, featureoutput, featureoutput)
        #print(x.shape)
        x = self.bn1(x) # torch.Size([5, 384, 33, 33])
        #print(x.shape)
        x = self.layer1(x) # ([5, 64, 33, 33])
        x = self.layer2(x) # [5, 128, 17, 17]
        x = self.layer3(x) # [5, 256, 9, 9]
        x = self.layer4(x) # [5, 512, 5, 5]
        x = F.avg_pool2d(x, 4) # ([5, 512, 1, 1])
        #print(x.shape)
        x = x.view(x.size(0), -1) # torch.Size([5, 512])
        #print(x.shape)
        x = self.linear(x)
        return x


class Integrated_Net(nn.Module):
    # TODO: integrate responses from 12 different frequency filters
    '''An upgrated version of spatial vision net, where boundary effect is alievated, positive and negative arctivities are
        separated, and backend is changed to a simpler architecture. '''

    def __init__(self, block, num_blocks, num_classes=1000, n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224,low_freq = False):
        self.inplanes = 64
        super(Integrated_Net, self).__init__()
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.imsize = imsize
        self.num_classes = num_classes

        if low_freq: # only employs the lower half of the spatial frequency filters,
            self.v1 = SV_integrated_net(n_freq  = int(self.n_freq/2), n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize, low_freq = True)
            self.conv1 = nn.Conv2d(2*int(self.n_freq/2)*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)
            self.n_freq = int(self.n_freq/2)
        else:
            self.v1 = SV_integrated_net(n_freq  = self.n_freq, n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize)
            self.conv1 = nn.Conv2d(2*n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)

        self.relu = nn.ReLU(inplace=True)
        # modify this part:
        self.in_planes = 2*self.n_freq*n_orient*n_phase
        self.bn1 = nn.BatchNorm2d(2*n_orient*n_phase)
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

        x = self.v1(x) # [batchsize,self.n_freq,positivity,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y]
        # TODO: Integrate different components of a frequency model so that during backpropagation, gradients that belong to different frequency parameters do not coadapt.
        n_freq = x.shape[1]
        x_total = np.zeros([x.shape[0],n_freq,self.num_classes]) # 1 is left for the gray value channel

        for i in range(0,n_freq):
            this_x = x[:,i,:,:,:,:,:]
            this_x = this_x.view(x.size[0],-1,x.size[-2],x.size[-1])
            this_x = self.bn1(this_x) # torch.Size([5, 384, 33, 33])
            this_x = self.layer1(this_x) # ([5, 64, 33, 33])
            this_x = self.layer2(this_x) # [5, 128, 17, 17]
            this_x = self.layer3(this_x) # [5, 256, 9, 9]
            this_x = self.layer4(this_x) # [5, 512, 5, 5]
            this_x = F.avg_pool2d(this_x, 4) # ([5, 512, 1, 1])
            this_x = this_x.view(this_x.size(0), -1) # torch.Size([5, 512])
            this_x = self.linear(this_x) # torch.Size([N, N_class])
            x_total[:,i,:] = this_x
        # TODO: return should be [N, N_freq, N_class]
        return x_total


class SpatialVisionNetIII(nn.Module):
    '''An extremely simplified version of SV_net; used for MNIST'''

    def __init__(self, num_classes=10, n_freq=12, n_orient=8, n_phase=2, imsize=32):
        super(SpatialVisionNetIII, self).__init__()
        self.inplanes = 64
        self.n_phase = n_phase
        self.imsize = imsize
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.v1 = SvNet(n_freq  = int(self.n_freq/2), n_orient = self.n_orient, n_phase = self.n_phase, imsize = self.imsize, low_freq = True)
        self.conv1 = nn.Conv2d(n_freq*n_orient*n_phase, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(64, 16, 2)
        self.fc1 = nn.Linear(16 * 3 * 3, num_classes)


    def forward(self, x):
        x = self.v1(x) # (n_batch,n_feature, featureoutput, featureoutput)
        x = self.pool(F.relu(self.conv1(self.dropout(x))))
        x = self.pool(F.relu(self.conv2(self.dropout(x))))
        x = x.view(-1, 16 * 3 * 3)
        x = self.fc1(self.dropout(x))
        return x


class MeanPadding(torch.nn.Module):
    # TODO: this function is intensive on memory, check if this is the most efficient way of doing mean padding. possibly use tensor.cat
    """
    pad each image with the mean of the pixel value, with their specified parameters,

    pad_l: padding length, this function assumes a squared equal padding on all sides of image
    sz: size of image, this function assumes a square image.
    """
    def __init__(self,pad_l,sz):
        super(MeanPadding, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.padding = torch.ones([1, 1, sz+pad_l*2, sz+pad_l*2]).to(device)
        self.pad_l = pad_l

    def forward(self, x):
        # n_img_per_batch = x.shape[0]  # this has to be true
        # print('n image per batch is ', n_img_per_batch)
        # print(x.shape[2],x.shape[3])
        assert x.shape[2] == x.shape[3], 'Warning: The image to be mean padded should be square sized'
        mean_batch = torch.mean(torch.mean(x, 3), 2).view([-1, 1, 1, 1])
        mean_batch = torch.mul(self.padding, mean_batch)
        mean_batch[:, :, self.pad_l:-self.pad_l, self.pad_l:-self.pad_l] = x
        return mean_batch


class LogGaborConvolution(torch.nn.Module):  # checked

    def __init__(self, sz=32, low_freq=False):
        """
        low_freq: only use half of the filters starting with the lowest frequency
        """
        super(LogGaborConvolution, self).__init__()
        pad_l = int(sz/2)  # pad half of the image size.
        self.low_freq = low_freq
        self.mean_padding = MeanPadding(pad_l, sz)  # padded with the mean of the image,
        self.combined_filters = self.load_filter_bank(low_freq=self.low_freq, sz=sz)
        # self.batchsize = batchsize

    def load_filter_bank(self, low_freq=False, sz=32):
        """Assumes a certain structure of filter file, returns a filter bank"""
        if sz == 32:  # load filter bank for 32 by 32 images
            path = './filter_data/spatial_filters_32_by_32.mat'
        elif sz == 128:
            path = './filter_data/spatial_filters_128_by_128.mat'
        else:  # load filter bank for 224 by 224 images
            path = './filter_data/spatial_filters_224_by_224.mat'
        # TODO: MAYBE TAKE THIS OUT OF THE EXECUTION,
        mat = scipy.io.loadmat(path)
        spatial_filters_imag = torch.tensor(mat['spatial_filters_imag'])
        spatial_filters_real = torch.tensor(mat['spatial_filters_real'])
        n_freq = spatial_filters_imag.shape[0]
        n_orient = spatial_filters_imag.shape[1]
        sz = spatial_filters_imag.shape[2]  # Image size
        n_filters = n_freq*n_orient*2  # multiply by phase
        # combine real and imaginary filters
        if not low_freq:
            banksize = n_filters
            start = 0
        else: 
            banksize = int(n_filters/2)
            start = int(n_freq/2)

        # modified code: 
        filter_banks = torch.zeros([banksize, 1, sz, sz])  # 1 is left for the gray value channel
        s = 0
        # TODO: make this more elegant.

        for f in range(start,n_freq):
            for o in range(0, n_orient):
                filter_banks[s, 0, :, :] = spatial_filters_real[f, o] - torch.mean(spatial_filters_real[f, o])
                s = s + 1
                filter_banks[s, 0, :, :] = spatial_filters_imag[f, o] - torch.mean(spatial_filters_imag[f, o])
                s = s + 1

        filter_banks = torch.FloatTensor(filter_banks)
        # filter_banks = filter_banks.type(torch.FloatTensor)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        filter_banks = filter_banks.to(device).contiguous()
        return filter_banks

    def forward(self, x):
        """
        convolve x with shape [n_imag_per_batch, 1, 2*imgsize, 2*imgsize]
        with hard coded filters with shape [2*n_freq*n_orient, 1, imgsize, img_size]
        returns a with shape [n_imag_per_batch,2*n_freq*n_orient,imgsize+1, img_size+1]
        """
        # assert self.batchsize == x.shape[0], "batch size needs to match the zeroth dimension of x"
        x = self.mean_padding(x).contiguous()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        x = F.conv2d(x, self.combined_filters)
        return x
    

# example module, skip for now. 
class Normalization(nn.Module):
    """Calculated Normalized Layer, essential the coefficients b_i,
    apply convolution with 4d gaussian with a^p, where a is the result of convolution with customized filter."""
    def __init__(self, n_freq=12, p=2, sz=224, l_x=32, l_y=32, l_f=3,l_o=3,padxy_l = 16,padxy_r = 15,padxy_t = 16,padxy_b = 15,pad_fo = 1,w_x = 1,
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
    def __init__(self, p=2.0,  q=1, C=0.25):
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
        r = torch.div(torch.pow(A, (self.p+self.q)), torch.add(torch.tensor(self.C**self.p), B))
        return r


# check the log gabor convolution works or not
class V1ImagenetNet(nn.Module):
    def __init__(self, n_freq=12, n_orient=8, n_phase=2, im_size=224, low_freq=False):
        super(V1ImagenetNet, self).__init__()
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.im_size = im_size
        self.conv_after_x = self.im_size*2 - self.im_size + 1
        self.conv_after_y = self.conv_after_x  # assume square images
        if low_freq:
            self.log_gabor = LogGaborConvolution(sz=self.im_size, low_freq=True)
        else:
            self.log_gabor = LogGaborConvolution(sz=self.im_size)
        self.sz_after_filtering = self.im_size*2 - self.im_size + 1
        self.normalization = Normalization(sz=self.sz_after_filtering)
        self.nonlinearity = Nonlinearity()

    def forward(self, a):
        batch_size = a.shape[0]
        a = self.log_gabor(a).contiguous()
        # B_normalization = self.normalization(a_) # the same till here
        # A = a_.view([batch_size,self.n_freq, self.n_orient, self.n_phase, self.conv_after_x, self.conv_after_y])
        a = self.nonlinearity(a, self.normalization(a)).contiguous()
        a = a.view([batch_size, -1, self.conv_after_x, self.conv_after_y])
        return a

class SvNet(nn.Module):
    """implement the spatial vision subcomponent of the module
    separate positive and negative subpart of the response before convolution"""
    def __init__(self, n_freq=12, n_orient=8, n_phase=2, imsize=224, low_freq=False):
        super(SvNet, self).__init__()
        self.imsize = imsize
        self.n_freq = n_freq
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.conv_after_x = self.imsize*2 - self.imsize + 1
        self.conv_after_y = self.conv_after_x  # assume square images
        if low_freq:    
            self.logabor = LogGaborConvolution(sz=self.imsize, low_freq=True)
        else:
            self.logabor = LogGaborConvolution(sz=self.imsize)

        self.sz_after_filtering = self.imsize*2 - self.imsize + 1
        self.normalization = Normalization(n_freq=self.n_freq, sz=self.sz_after_filtering)
        self.nonlinearity = Nonlinearity()

    def sign_segragation(self, this_filter_o):
        # input: filter response
        # output, positive and negative filter activities, to be processed separately
        sign = this_filter_o > 0
        if torch.cuda.is_available():
            sign = sign.type(torch.cuda.FloatTensor)
        else:
            sign = sign.type(torch.FloatTensor)

        positive = this_filter_o*sign
        negative = -1*this_filter_o*(1-sign)
        return positive, negative  # returns only positive values for competition/gain control

    def forward(self, images):
        # separate positive and negative part of a_ for normalization
        batchsize = images.shape[0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        a_ = self.logabor(images).contiguous()
        a_pos, a_neg = self.sign_segragation(a_)
        # B_normalization_pos = self.normalization(a_pos) # the same till here
        # B_normalization_neg = self.normalization(a_neg) # the same till here
        a_pos = a_pos.view([batchsize, self.n_freq, self.n_orient, self.n_phase, self.conv_after_x, self.conv_after_y])
        a_neg = a_neg.view([batchsize, self.n_freq, self.n_orient, self.n_phase, self.conv_after_x, self.conv_after_y])
        a_pos = self.nonlinearity(a_pos, self.normalization(a_pos)).contiguous()
        a_neg = self.nonlinearity(a_neg, self.normalization(a_neg)).contiguous()
        a_pos = a_pos.view([batchsize, -1,self.conv_after_x, self.conv_after_y])
        a_neg = a_neg.view([batchsize, -1, self.conv_after_x, self.conv_after_y])
        R = torch.cat((a_pos, a_neg), 1)
        # [n_img_per_batch, 192*2, 32, 32]
        return R



class SV_integrated_net(nn.Module):
    '''implement the spatial vision subcomponent of the module
    separate positive and negative subpart of the response before convolution'''
    def __init__(self,n_freq  = 12, n_orient = 8, n_phase = 2, imsize = 224, low_freq = False):
        super(SV_integrated_net, self).__init__()
        self.imsize = imsize
        self.n_freq = n_freq
        #print('self.n_freq',self.n_freq)
        self.n_orient = n_orient
        self.n_phase = n_phase
        self.conv_after_x = self.imsize*2 - self.imsize + 1
        self.conv_after_y = self.conv_after_x # assume square images
        if low_freq:
            self.logabor = LogGaborConvolution(sz=imsize, low_freq=True)
        else:
            self.logabor = LogGaborConvolution(sz=imsize)

        self.sz_after_filtering = self.imsize*2 - self.imsize + 1
        self.normalization = Normalization(n_freq=self.n_freq, sz=self.sz_after_filtering)
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

        # R_pos = R_pos.view([batchsize,-1,self.conv_after_x,self.conv_after_y])
        # R_neg = R_neg.view([batchsize,-1,self.conv_after_x,self.conv_after_y])
        # R = torch.cat((R_pos,R_neg),1)
        R = torch.stack((R_pos,R_neg),2) # [batchsize,self.n_freq,positivity,self.n_orient,self.n_phase,self.conv_after_x,self.conv_after_y]
        # [n_img_per_batch, 192*2, 32, 32]

        return R



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes) # inchannel = inplanes, outchannel = planes
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


def construct_svnet_1(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO: construct pre-trained code
    model = SpatialVisionNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.to(device)
    return model


def construct_svnet_low_f(**kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpatialVisionNet(BasicBlock, [2, 2, 2, 2], low_freq=True, **kwargs)
    model.to(device)
    return model


def construct_svnet_2(**kwargs):
    # a net with positive and negative signal separation and a simple backend.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpatialVisionNetII(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.to(device)
    return model


def construct_svnet_2_low_f(**kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpatialVisionNetII(BasicBlock, [2, 2, 2, 2], low_freq=True, **kwargs)
    model.to(device)
    return model


def construct_integrated_net(**kwargs):
    # a neural network combining all different frequencies.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Integrated_Net(BasicBlock, [2, 2, 2, 2], **kwargs)
    model.to(device)
    return model


def construct_simple_net_3(pretrained=False, **kwargs):
    # designed for MNIST experiments to test SV_net's adversarial robustness.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SpatialVisionNetIII(**kwargs)
    model.to(device)
    return model


def construct_resnet18(pretrained=False, **kwargs):
    """Constructs a Vanilla ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained: # loading state dictionary from online models. Should not be used here.
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def construct_ln_net():
    """Construct Versions of SV Net where Divisive Normalization is learned"""
    pass



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
