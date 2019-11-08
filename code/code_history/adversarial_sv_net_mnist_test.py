# try sv net on MNIST
import torch
import torchvision
import model as models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


n_epochs = 3
batch_size_train = 1
batch_size_test = 1
learning_rate = 0.01
momentum = 0.001
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# train loader and test loader for training the network, but not for adversarial attacjs.
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=False,transform=torchvision.transforms.Compose([
                           torchvision.transforms.Pad(2, fill=0, padding_mode='constant'),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
    (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Pad(2, fill=0, padding_mode='constant'),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


# network = models.simple_net_III(num_classes=10, batchsize = batch_size_train, n_orient = 8, n_phase = 2, imsize = 32)
network = models.simple_net_III(num_classes=10, n_orient = 8, n_phase = 2, imsize = 32)

optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
criterion = torch.nn.CrossEntropyLoss()


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*4) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), 'mnist_sv_net_model.pth')
            torch.save(optimizer.state_dict(), 'mnist_sv_net_model_optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            total = total + 4
            i = i + 4
            if i > 100:
                break
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set, Accuracy: {}/{} ({:.0f}%)\n'.format( correct, total,100. * correct / total))


network.load_state_dict(torch.load('mnist_sv_net_model_bs1.pth'))




# innocent image loader




# loader for adversarial fool box use
atrain_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=False,transform=torchvision.transforms.Compose([
                           torchvision.transforms.Pad(2, fill=0, padding_mode='constant'),
                           torchvision.transforms.ToTensor(),
                             ])),
  batch_size=batch_size_train, shuffle=True)

atest_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Pad(2, fill=0, padding_mode='constant'),
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=batch_size_test, shuffle=True)


# loader for using the network to evaluate(getting back to the network training stage)
transform_back =  torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))])

import foolbox
import torch
import torchvision.models as m
import numpy as np


examples = enumerate(atest_loader)

test_network = network.eval()  # for CPU, remove cuda()
fmodel = foolbox.models.PyTorchModel(test_network, bounds=(0.0, 1.0), num_classes=10,
                                     preprocessing=(0.1307, 0.3081))
# this_image = mnist.test.images[i]
# adversarial_L2BasicIterative = L2BasicIterative(this_image, label=np.argmax(mnist.test.labels[i]))
attack = foolbox.attacks.BoundaryAttack(fmodel)

''' Test on 10000 number of images'''

# eps= np.load('SV_net_adversarial_mnist.npy')

# for J in range(0,100):
batch_idx, (example_data, example_targets) = next(examples)
this_image, this_label = example_data, example_targets
# regular image, without normalization.
# for evaluation of the original network
image_for_evaluation = torch.zeros(this_image.shape)
for i in range(0, batch_size_train):
    image_for_evaluation[i, :, :, :] = transform_back(torch.FloatTensor(this_image[i, :, :, :]))
# instantiate the model
this_image = np.array(this_image)
## this needs to be modified with the network
pred = np.argmax(fmodel.forward((this_image)), axis=1)
# output = network(image_for_evaluation)
# pred = output.data.max(1, keepdim = True)[1]
# print('predicted class', pred)
# print('True Label', this_label)
print('attack is starting')

adversarial = attack(np.array(this_image), np.array(this_label), verbose = True)
if adversarial is not None:
    diff = np.sqrt(np.mean(np.square(this_image - adversarial)))
    # eps[-J] = diff
    print(diff)
# else:
    # eps[-J] = 1000
# if J % 100 == 0:
#     print(J)

# np.save('SV_net_adversarial_mnist.npy',eps)

import numpy as np
# print(epss[9090:9999])


# n_pixel = 28*28
# for i in range(0,100):
#     this_image = this_example
#     adversarial = attack(np.array(image), np.array(label))
#     diff = np.abs(this_image - adversarial)**2/n_pixel
#
print(this_image.shape)
print(adversarial.shape)
preda = np.argmax(fmodel.forward((adversarial)),axis = 1)
plt.subplot(211)
plt.imshow(this_image[0,0,:,:])
plt.title(pred)
plt.subplot(212)
plt.imshow(adversarial[0,0,:,:])
plt.title(preda)

plt.savefig('adversarial_example11.png')

