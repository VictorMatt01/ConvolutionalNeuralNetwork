#Author: Victor Matthijs
#Date: 07/12/2018
#This little project is a test project: how to write a CNN
#The code below is based on the lessons and material that I get from the Udacity cours:
#PyTorch Scholarship Challenge from Facebook

import torch
import numpy as np 

# we will check of the computer/workstations has CUDA available
train_CNN_on_GPU = torch.cuda.is_available()
if train_CNN_on_GPU:
    # there is a GPU available
    print("we need to use extern GPUs")

# else we must use an other option to get more computing power to train or model


# STEP 1: load the Data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

number_of_subprocesses = 0
#how many samples per batch
batch_size = 20
#the percentage to use as validation of our trainingset
validation_size = 0.2

#  we first need to transform our data into a Tensor(pytorch array)
# this link can help to understand the fucntions we used below: https://pytorch.org/docs/stable/torchvision/transforms.html 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

#we now choose the training and test datasets out of our data
# you can find other datasets info if you click this link: https://pytorch.org/docs/stable/torchvision/datasets.html 
train_data = datasets.CIFAR10('data', train = True, download = True, transform = transform) 
test_data = datasets.CIFAR10('data', train = False, download = True, transform = transform)

# we will now obtain our training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(validation_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# we will define some samplers to obtain training batches
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(valid_idx)

#we prepare the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler, num_workers = number_of_subprocesses)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = validation_sampler, num_workers = number_of_subprocesses)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers = number_of_subprocesses)

#we specify the output classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

#STEP 2: visualize a batch of training data, this is optional
import matplotlib.pyplot as plt
"""
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

plt.show()
"""

#STEP 3: define our Network Architecture
import torch.nn as nn
import torch.nn.functional as F 

# we define our CNN architecture here
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #first convolutional layer (input is 32x32x3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        #second convolutional layer (input 16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        #third convolutional layer (input 8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        #max pooling layer
        self.pool = nn.MaxPool2d(2,2)

        #first linear layer
        self.fc1 = nn.Linear(64*4*4, 500)
        #second linear layer
        self.fc2 = nn.Linear(500, 200)
        #third linear layer
        self.fc3 = nn.Linear(200, 10)

        #dropout function
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        #we flatten the image inout with the function view
        x = x.view(-1, 64*4*4)

        #add dropout layer
        x = self.dropout(x)

        #add first hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #add second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        #add third hidden layer
        x = F.relu(self.fc3(x))

        return x

model = Net()
#we print out our model
print(model)

# we also need to specify a loss function and an Optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

#STEP 4: Training the network
#important: make sure we don't have overfitting in our CNN
#we will save the CNN when the validation loss starts to increas

n_epochs = 5 #You can choose a number of epochs

valid_loss_min = np.Inf #with this we track changes in our validation loss

for epoch in range(1, n_epochs+1):
    #we train our model in this loop
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
        #if the CUDA is available then move the tensor to the GPU
        if train_CNN_on_GPU:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        #forward pass: we compute the predicted ouputs
        output = model(data)
        #we calculate the batch loss
        loss = criterion(output, target)
        #backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        #we update the parameters
        optimizer.step()
        #and as last update the training loss
        train_loss += loss.item()*data.size(0)
    
    model.eval()
    for data, target in valid_loader:
        if train_CNN_on_GPU:
            data, target = data.cuda(), target.cuda()
        #forward pass
        output = model(data)
        #calculate the batch loss
        loss = criterion(output, target)
        #update the validation_loss
        valid_loss += loss.item()*data.size(0)
    
    #we calculate the average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    #we print out the training and validation losses for control
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

#After we looped through the number op epochs we get our latest saved model and start to test it
model.load_state_dict(torch.load('model_cifar.pt'))

#STEP 5: Test the CNN
#to be continued!!!
