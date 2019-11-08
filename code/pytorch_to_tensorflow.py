# TODO: Convert from pytorch file to IR representation
# TODO: Then convert the file into Tensorflow
# TODO: Test it on CIFAR Images
# TODO: Then put it on trained IMAGENET images

$ mmdownload -f pytorch -h
Support frameworks: ['alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

$ mmdownload -f pytorch -n resnet101 -o ./
Downloading: "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth" to /home/ruzhang/.torch/models/resnet101-5d3b4d8f.pth
███████████████████| 102502400/102502400 [00:06<00:00, 15858546.50it/s]
PyTorch pretrained model is saved as [./imagenet_resnet101.pth].