import torch
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from DatasetCreator import HAVSDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

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

def display_random_errors(img_errors, pred_errors, obs_errors, dataset):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 5
    errors_index = random.sample(range(0, len(img_errors)), nrows*ncols)
    classes = dataset.enc.classes_.tolist()
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize=(14,10))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((128,44)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format( classes[int(pred_errors[error])], classes[int(obs_errors[error])]  ))
            n += 1

def train_model(num_epochs, train_loader, val_loader, criterion, optimizer, device, model_on_device):
    start = time.time()
    optimizer_name, optimizer_params_dict = get_info(optimizer)
    writer = SummaryWriter(comment=f"_model_{model_on_device._get_name()}_criterion_{criterion._get_name()}_optimizer_{optimizer_name}")
    
    num_batches = len(train_loader)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
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

        
        test_loss, test_accuracy , _, _, _ = evaluate_model(val_loader, device, model_on_device, criterion)

        writer.add_scalar('Loss/Train', avg_epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_epoch_accuracy, epoch)
        writer.add_scalar('Loss/Validation', test_loss, epoch)
        writer.add_scalar('Accuracy/Validation', test_accuracy, epoch)
        
        print('Epoch: {}/{} \t Training Loss: {:.4f}, Accuracy: {:.2f}, Testing Loss: {:.4f}, Accuracy: {:.2f}'.format(epoch, num_epochs, train_loss, accuracy, test_loss, test_accuracy))

        
        early_stopping(test_loss, model_on_device)
        if early_stopping.early_stop:
            print("Early stopping")
            break    
       
    ## Please revisit this error in names -> test/val
    stop = time.time()
    duration_s = stop - start
    writer.add_text('optimzer_parameters', str(optimizer_params_dict))
    writer.add_text('model', str(model_on_device))
    writer.add_text('Duration_s', str(duration_s))

    dummy_data =  torch.randn(256, 1, 128 ,45)
    dummy_data = dummy_data.to(device, dtype=torch.float)
    writer.add_graph(model=model_on_device, input_to_model=dummy_data)

    writer.close()

def evaluate_model(loader, device, model_on_device, criterion, *args):

    """Evaluate Performance on test set"""
    model_on_device.eval() # Turn off gradient computations
    num_batches = len(loader)
    correct = 0
    total = 0
    running_loss = 0
    y_tot=torch.empty(0)
    y_pred_tot=torch.empty(0)
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


            labels=labels.cpu()
            predicted=predicted.cpu()

            y_tot = torch.cat((y_tot, labels), 0)
            y_pred_tot = torch.cat((y_pred_tot, predicted), 0)

    accuracy = 100 * correct / total       
    accuracy = 100 * correct / total       
    errors = (y_pred_tot - y_tot  != 0)
    y_pred_errors = y_pred_tot[errors]
    y_true_errors = y_tot[errors]
    # Plotting the Confusion Matrix
    assert len(args)==2 or len(args)==0, 'Please insert both dataset and dataset name'
    if args:
        cm = confusion_matrix(y_tot.numpy(), y_pred_tot.numpy())
        np.set_printoptions(precision=4)

        # Coloured confusion matrix
        plt.figure(figsize = (12,12))
        cm = confusion_matrix(y_tot.numpy(), y_pred_tot.numpy(), normalize="true")
        plt.imshow(cm, cmap=plt.cm.Blues)

        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        
        plt.xticks(range(6))
        plt.yticks(range(6))
        plt.xlabel("Prediction")
        plt.ylabel("True")

        # We can retrieve the categories used by the LabelEncoder
        classes = args[0].enc.classes_.tolist()
        plt.gca().set_xticklabels(classes)
        plt.gca().set_yticklabels(classes)

        plt.title("Normalized Confusion Matrix For "+ args[1] + " Data")
        plt.colorbar()
        plt.show()
    return running_loss / num_batches, accuracy, errors, y_pred_errors, y_true_errors 


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

def createDataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    # Define the transforms
    transform = transforms.Compose(
        [
        transforms.ToTensor()
        ])
    # Create the datasets
    train_dataset = HAVSDataset(x_train, y_train, transform=transform)
    val_dataset = HAVSDataset(x_val, y_val, transform=transform)
    test_dataset = HAVSDataset(x_test, y_test, transform=transform)

    # Creating the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader