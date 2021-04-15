import torch
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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

def train_model(num_epochs, train_loader, val_loader, criterion, optimizer, device, model_on_device):
    optimizer_name, optimizer_params_dict = get_info(optimizer)
    writer = SummaryWriter(comment=f"_model_{model_on_device._get_name()}_criterion_{criterion._get_name()}_optimizer_{optimizer_name}")
    num_batches = len(train_loader)
    for epoch in range(1, num_epochs+1):
        model_on_device.train() # Turn on Dropout, BatchNorm etc
        train_loss_per_batch = np.empty(num_batches)
        accuracy_per_batch = np.empty(num_batches)
        train_loss = 0
        correct = 0
        total = 0
        accuracy = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
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
            accuracy = 100 * correct / total

            train_loss_per_batch[batch_idx] =  train_loss
            accuracy_per_batch[batch_idx] = accuracy

        avg_epoch_train_loss = np.mean(train_loss_per_batch)
        avg_epoch_accuracy = np.mean(accuracy_per_batch)

        test_loss, test_accuracy = evaluate_model(val_loader, device, model_on_device, criterion)

        writer.add_scalar('Loss/Train', avg_epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_epoch_accuracy, epoch)
        writer.add_scalar('Loss/Validation', test_loss, epoch)
        writer.add_scalar('Accuracy/Validation', test_accuracy, epoch)
        
        print('Epoch: {}/{} \t Training Loss: {:.4f}, Accuracy: {:.2f}, Testing Loss: {:.4f}, Accuracy: {:.2f}'.format(epoch, num_epochs, train_loss, accuracy, test_loss, test_accuracy))
    ## Please revisit this error in names -> test/val

    writer.add_text('optimzer_parameters', str(optimizer_params_dict))
    writer.add_text('model', str(model_on_device))
    writer.close()

def evaluate_model(loader, device, model_on_device, criterion):
    """Evaluate Performance on test set"""
    model_on_device.eval() # Turn off gradient computations
    num_batches = len(loader)
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device).float()
            labels = labels.to(device)
            outputs = model_on_device(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

    accuracy = 100 * correct / total       

    return running_loss / num_batches, accuracy 


def get_class_weights(y_train, device):
    """Get the class weight to manage data imbalance"""
    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(y_train),
                                                    y_train)
    class_weights = torch.from_numpy(class_weights).float().to(device)

    return class_weights

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

def get_train_test_data(compressed_file_path):
    # Extracting data from the compressed file

    processed_data = np.load(compressed_file_path) # Unzipping
    x_train = processed_data["x_train"]
    x_test = processed_data["x_test"]
    x_val = processed_data["x_val"] 
    y_train = processed_data["y_train"]
    y_test = processed_data["y_test"]
    y_val = processed_data["y_val"]

    return (x_train, x_test, x_val, y_train, y_test, y_val)

def get_info(optimizer):
    optimizer_name = optimizer.__class__.__name__
    optimizer_params_dict = {key:group[key] for group in (optimizer.param_groups) for key in sorted(group.keys()) if key != 'params'}

    return optimizer_name, optimizer_params_dict