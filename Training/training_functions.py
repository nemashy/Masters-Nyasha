import torch
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    """Checks the device that cuda is running on"""

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")

    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    return device

def train_cae(num_epochs, train_loader, criterion, optimizer, device, model_on_device):
    """Trains the CAE with the input data as target"""

    for epoch in range(1, num_epochs+1):
        # monitor training loss
        train_loss = 0.0
        for images, _ in train_loader: 
            # _ stands in for labels, here
            # no need to flatten images
            images = images.to(device, dtype=torch.float)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model_on_device(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            )
            )

def train_model(num_epochs, train_loader, criterion, optimizer, device, model_on_device):

    for epoch in range(1, num_epochs+1):
        train_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device)
            output = model_on_device(images)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()/images.shape[0]
        
        print('Epoch: {}/{} \t Training Loss: {}, Accuracy: {}'.format(epoch, num_epochs, train_loss, 100 * correct / total))


def eval(test_loader, device, model_on_device):
    """Evaluate Performance on test set"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device).float()
            labels = labels.to(device)
            outputs = model_on_device(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total       
    print('Accuracy of the network on the test images: %d %%' % (
        accuracy))

    return accuracy


def get_class_weights(y_train, device):
    """Get the class weight to manage data imbalance"""
    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(y_train),
                                                    y_train)
    class_weights = torch.from_numpy(class_weights).float().to(device)

    return class_weights

def nyasha():
    pass


def visualise_cae_performance(data_loader, device, model_on_device, batch_size, img_width, img_height):
    """Visualise how the current CAE model is performing"""
    #Batch of test images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.to(device, dtype=torch.float)

    #Sample outputs
    output = model_on_device(images)
    images = images.cpu().numpy()

    output = output.view(batch_size, 1, img_width, img_height)
    output = output.cpu().detach().numpy()

    #Original Images
    print("Original Images")
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(np.squeeze(images[idx]))
        ax.set_title(labels[idx])
    plt.show()

    #Reconstructed Images
    print('Reconstructed Images')
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
    for idx in np.arange(5):
        ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(np.squeeze(output[idx]))
        ax.set_title(labels[idx])
    plt.show()