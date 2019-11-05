import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


cifar_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.4734,), (0.252,))
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])


# cifar10
train_set = datasets.CIFAR10(root='cifar/', train=True, download=False, transform=cifar_transform)
testset = datasets.CIFAR10(root='cifar/', train=False, download=False, transform=cifar_transform)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epochs):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[{}, {}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

def get_image_svd(num_batches_to_use, window_size):
    vecs=[]    
    for i, data in enumerate(trainloader,1):
        inputs, label = data
        inputs = inputs.squeeze()
        imsize = inputs.shape[1:]
        for ig in range(inputs.shape[0]): # loops images
            for r in range(imsize[0]-window_size[0]+1):
                for c in range(imsize[1]-window_size[1]+1):
                    vecs.append(inputs[ig,r:r+window_size[0], c:c+window_size[1]].reshape(-1))
        
        if i == num_batches_to_use:
            break
    a= np.vstack(vecs).T
    u,s,v = np.linalg.svd(a)
    return u,s,v
# print(train_set.data.shape)
# print(train_set.data.mean(axis=(0,1,2))/255)
# print(train_set.data.std(axis=(0,1,2))/255)
# # (50000, 32, 32, 3)
# # [0.49139968  0.48215841  0.44653091]
# # [0.24703223  0.24348513  0.26158784]



if __name__ == "__main__":
    #train(2)
    get_image_svd(3,(6,5))
    #print(iter(trainloader).next()[0].shape)

    # print(train_set.data.shape)
    # print(train_set.data.mean(axis=(0,1,2,3))/255)
    # print(train_set.data.std(axis=(0,1,2,3))/255)

# tensor([[-1.8553, -1.8391, -1.8230, -1.8391],
#         [-1.8391, -1.8230, -1.8068, -1.8230],
#         [-1.8230, -1.8230, -1.8230, -1.8230],
#         [-1.8230, -1.8230, -1.8391, -1.8230]])
