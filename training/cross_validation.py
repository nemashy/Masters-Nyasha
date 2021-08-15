import os
import torch
from torch import nn
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from CNN import ErnNet
from model_training import EarlyStopping
from DatasetCreator import HAVSDataset
from model_training import ModelTrainer

def reset_weights(m):
    '''
        Try resetting model weights to avoid
        weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def reset_optimizer(optimizer, lr):
    optimizer.params_groups[0]['lr'] = lr
    return optimizer

class TrainTestData:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def get_dataset(self):
        transform = transforms.Compose(
        [
        transforms.ToTensor()
        ])
        return HAVSDataset(self.x_data, self.y_data, transform=transform)

class CrossValidation():
    def __init__(self, network, train_data, test_data, device, criterion, num_folds):
        self.train_data = train_data
        self.test_data = test_data
        self.network = network
        self.device = device
        self.criterion = criterion
        self.num_folds = num_folds
        self.classes = None

    def train_model(self, num_epochs, batch_size, scheduler=True):
        # Set fixed random number seed
        torch.manual_seed(42)

        # For fold results
        results = {}

        # Create the datasets
        train_dataset = self.train_data.get_dataset()

        # Saving labels
        self.classes = train_dataset.enc.classes_.tolist()

        # Define the K-fold Cross Validator
        kfold = StratifiedKFold(n_splits=self.num_folds, shuffle=False)

        # Start print
        print('--------------------------------')

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset, self.train_data.y_data)):

            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')



            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                                train_dataset, 
                                batch_size=batch_size, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                                train_dataset,
                                batch_size=batch_size, sampler=test_subsampler)

            # Init the neural network
            self.network = ErnNet()
            network_on_device = self.network.to(self.device) # Move model to the current device
            network_on_device.apply(reset_weights)


            lr = 1e-4
            transition_steps = [10,20,30,40,50,60]
            gamma = 0.8
            optimizer = optim.Adam(network_on_device.parameters(), lr)


            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=20, verbose=True)

            # Reset the learning rate

            scheduler =  MultiStepLR(
            optimizer,
            transition_steps,
            gamma=gamma,
            last_epoch=-1,
            verbose=True,
        )
            
            # Run the training loop for defined number of epochs
            for epoch in range(0, num_epochs):

                # Print epoch
                print(f'Starting epoch {epoch+1}')

                # Set current loss value
                current_loss = 0.0

                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):
                
                    # Get inputs
                    inputs, targets = data
                    inputs = inputs.to(self.device, dtype=torch.float)
                    targets = targets.to(self.device)
        
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Perform forward pass
                    outputs = network_on_device(inputs)
                    
                    # Compute loss
                    loss = self.criterion(outputs, targets)
                    
                    # Perform backward pass
                    loss.backward()
                    
                    # Perform optimization
                    optimizer.step()
                    
                    # Print statistics
                    current_loss += loss.item()
                    if i % 500 == 499:
                        print('Loss after mini-batch %5d: %.3f' %
                                (i + 1, current_loss / 500))
                        current_loss = 0.0

                # Check early stopping
                test_loss, accuracy, _, _, _ = self.evaluate_model(network_on_device, testloader)

                if scheduler:
                    scheduler.step()

                early_stopping(test_loss, network_on_device)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break           
            # Process is complete.
            print('Training process has finished. Saving trained model.')

            # Print about testing
            print('Starting testing')

            # Load last checkpoint (best results)
            self.network.load_state_dict(torch.load("checkpoint.pt"))
            network_on_device = self.network.to(self.device)

            # Saving the best model
            save_path = f'./model-fold-{fold}.pth'
            torch.save(network_on_device.state_dict(), save_path)

            # Print accuracy
            _, accuracy, _, _, _ = self.evaluate_model(network_on_device, testloader, show_cm=True)
            print('Accuracy for fold %d: %d %%' % (fold, accuracy))
            print('--------------------------------')
            results[fold] = accuracy

            # Print fold results
            print(f'K-FOLD CROSS VALIDATION RESULTS FOR {self.num_folds} FOLDS')
            print('--------------------------------')
            sum = 0.0
            for key, value in results.items():
                print(f'Fold {key}: {value} %')
                sum += value
            print(f'Average: {sum/len(results.items())} %')

    def evaluate_model(self, model_on_device, data_loader, show_cm=False):

        """Evaluate Performance on test set"""
        model_on_device.eval()  # Turn off gradient computations
        num_batches = len(data_loader)
        correct = 0
        total = 0
        running_loss = 0
        y_tot = torch.empty(0)
        y_pred_tot = torch.empty(0)

        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images = images.to(self.device).float()
                labels = labels.to(self.device)
                outputs = model_on_device(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

                labels = labels.cpu()
                predicted = predicted.cpu()

                y_tot = torch.cat((y_tot, labels), 0)
                y_pred_tot = torch.cat((y_pred_tot, predicted), 0)

        accuracy = 100 * correct / total
        accuracy = 100 * correct / total
        errors = y_pred_tot - y_tot != 0
        y_pred_errors = y_pred_tot[errors]
        y_true_errors = y_tot[errors]

        # Plotting the Confusion Matrix
        if show_cm:
            ModelTrainer.generate_confusion_matrix(self.classes, y_tot, y_pred_tot)

        return running_loss / num_batches, accuracy, errors, y_pred_errors, y_true_errors
