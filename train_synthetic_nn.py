import numpy as np
from numpy import random
from numpy.linalg import svd
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.special import softmax


def create_class_assignments(n_samples, n_class):
    class_assignments = (random.rand(n_samples-n_class)*n_class).astype(int)
    class_assignments = np.hstack([class_assignments,np.arange(n_class,dtype=int)])
    random.shuffle(class_assignments)
    return class_assignments
    

def network(a, b, f, b2, c):
    def eval_network(x):
        x = np.tanh(np.matmul(a,x))
        x = np.tanh(np.matmul(b,x)+f)
        x = np.tanh(b2 * x)
        #x = np.tanh(np.matmul(b2,x))
        x = softmax(np.matmul(c,x))
        return x
    return eval_network

def generate_layers_and_samples(n_dims, n_class, seed=1234):
    # x = np.tanh(np.matmul(a,x))
    # 
    random.seed(seed)
    a, a_inv = random_transform_matrix(n_dims)
    b = orthant_detector_layer(n_dims)
    n_samples = 2**n_dims
    f = -0.25 *np.ones((1,n_samples))
    b2 = np.ones(n_samples)
    class_assignments = create_class_assignments(n_samples,n_class)    
    c = class_assignment_layer(class_assignments)
    # c = (1/np.sum(c,axis=1)[:,np.newaxis])*c
    samples = random.rand(n_samples, n_dims)
    samples = samples*b
    samples = np.arctanh(samples)
    samples = np.matmul(a_inv, samples.T)
    return a*500, b*0.05, f, b2*18, c, class_assignments, samples
    
def class_assignment_layer(class_assignments):
    # create a layer that produces the maximum value for the correct class
    n_classes = max(class_assignments)+1
    n_samples = len(class_assignments)
    l = np.tile(class_assignments,(n_classes,1)).astype(np.float)
    for i in range(n_classes):
        l[i,:] = l[i,:]== i
    l = (1/np.sum(l,axis=1)[:,np.newaxis])*l   
    #l[l==0] = -0.08
    return l

class Net(nn.Module):
    def __init__(self, dims, n_classes):
        super(Net, self).__init__()
        
        self.dims = dims
        self.n_classes = n_classes
        
        self.a = nn.Linear(dims, dims, bias=False)
        self.b = nn.Linear(dims, 2 ** dims, bias=False)
        self.b2 = torch.ones(2 ** dims)
        self.f = torch.zeros(2 ** dims)
        self.c = nn.Linear(2 ** dims, n_classes, bias=False)

        
    def assign_values_to_layers(self, **kwargs):
        attrs_to_set = set(['a','b','c']).intersection(kwargs.keys())
        for attr in attrs_to_set:
            getattr(self, attr).weight = torch.nn.Parameter(torch.Tensor(kwargs[attr]))
            #print(attr, getattr(self, attr).weight.shape) #
        if 'f' in kwargs:
            self.f = torch.nn.Parameter(torch.Tensor(kwargs['f']))
            attrs_to_set.add('f')
        if 'b2' in kwargs:
            self.b2 = torch.nn.Parameter(torch.Tensor(kwargs['b2']))
            attrs_to_set.add('b2')
        print('attrs updated: {}'.format(attrs_to_set))

    def forward(self, x):
        x = torch.tanh(self.a(x))
        x = torch.tanh(self.b(x)+ self.f)
        x = torch.tanh(x * self.b2 )
        x = self.c(x)
        return x


def train(net, trainloader, epochs):
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
            if i % 50 == 49:    # print every 200 mini-batches
                print('[{}, {}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def random_transform_matrix(d):
    u,w,_ = svd(random.rand(d,d))
    return np.matmul(u,np.diag(w)), np.matmul(np.diag(0.2/w),u.T)


def fill_uniform_random(d):
    # create a D-dimensional hypercube in the range (-1,1) in each axis
    # and place one sample randomly in each bin
    samples = 2**d
    coords = random.rand(samples,d)
    sign_lists= []
    for r in range(d-1,-1,-1):
        sign_lists.append(([0] * 2**r + [1] * 2**r) * 2**(d-r-1))
    
    sign_arr = np.array(sign_lists).T*2-1
    return coords * sign_arr, sign_arr


def orthant_detector_layer(d):
    # create a D-dimensional hypercube in the range (-1,1) in each axis
    # and place one sample randomly in each bin
    sign_lists= []
    for r in range(d-1,-1,-1):
        sign_lists.append(([0] * 2**r + [1] * 2**r) * 2**(d-r-1))
    
    sign_arr = np.array(sign_lists).T*2-1
    return sign_arr



def main(net = None, samples=None, class_assignments=None):
    if type(net)==int:
        seed = net
        net = None
    else:
        seed = 1234
    if net is None:
        net = Net(6, 8)
        a, b, f, b2, c, class_assignments, samples = generate_layers_and_samples(6, 8, seed)
        net.assign_values_to_layers(a=a, b= b, f=f, b2=b2, c= c)

    tensor_samples = torch.Tensor(samples.T)
    correct = sum(np.argmax(net(tensor_samples).detach().numpy(), axis=1)==class_assignments)
    print("correct before training: {}".format(correct))
    trainloader = torch.utils.data.DataLoader(list(zip(tensor_samples, class_assignments)), batch_size=1, shuffle=True, num_workers=0)
    train(net, trainloader, 100)
    correct = sum(np.argmax(net(tensor_samples).detach().numpy(), axis=1)==class_assignments)
    print("correct after training: {}".format(correct))
    return net


if __name__ == "__main__":
    main()
