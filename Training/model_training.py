from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import MultiStepLR

def get_device():
    """Checks the device that cuda is running on"""

    if torch.cuda.is_available():
        device = torch.device(
            "cuda:0"
        )  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")

    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    return device

class Scheduler:
    def __init__(self, optimizer, transition_steps, gamma_value):
        self.optimizer = optimizer
        self.transition_steps = transition_steps
        self.gamma_value = gamma_value

    def get_MultiStepLR(self):
        return MultiStepLR(
            self.optimizer,
            self.transition_steps,
            gamma=self.gamma_value,
            last_epoch=-1,
            verbose=True,
        )
        # Introducing the learning rate schedule
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,20,30,40,50,60] , gamma=0.8, last_epoch=-1, verbose=True)
        # MultiStepLR(optimizer, [4, 10, 15, 20, 25, 30, 35, 40, 45, 50], gamma=0.5, last_epoch=-1, verbose=False)
        # [5, 15, 20, 25, 30, 35, 40, 45, 50]
        # StepLR(optimizer, step_size=10, gamma=0.2, last_epoch=-1, verbose=False)
        # initialize the early_stopping object

class HAVSDataset(Dataset): # Human Activity, Vehicle and Sphere (HAVS)
    """Create a custom dataset"""

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.transform = transform
        self.enc = LabelEncoder()
        targets = self.enc.fit_transform(targets.reshape(-1,))
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index): # Memory efficient way of getting items
        if torch.is_tensor(index):
            index = int(index.item())
            
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.data[index]
            x = self.transform(x) # Convert numpy.ndarray to pytorch tensor

        return x, y

    def __len__(self):
        return len(self.data)

class ModelTrainer:
    def __init__(self, model, criterion, data_loaders_and_classes):
        self.model = model
        self.device = get_device()
        self.model_on_device = model.to(self.device)

        self.train_loader = data_loaders_and_classes['train_loader']
        self.val_loader = data_loaders_and_classes['val_loader']
        self.classes = data_loaders_and_classes['classes']
        self.criterion = criterion
        
        self.compressed_model = None
        self.scheduler = None

    def train_model(self, num_epochs, optimizer):
        
        writer = SummaryWriter()
        _, optimizer_params_dict = get_info(optimizer)
        num_batches = len(self.train_loader)
        early_stopping = EarlyStopping(patience=20, verbose=True)

        start = time.time()
        for epoch in range(1, num_epochs + 1):
            self.model_on_device.train()  # Turn on Dropout, BatchNorm etc
            train_loss_per_batch = np.empty(num_batches)
            accuracy_per_batch = np.empty(num_batches)
            train_loss = 0
            correct = 0
            total = 0
            accuracy = 0

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device, dtype=torch.float)
                labels = labels.to(self.device)

                output = self.model_on_device(images)
                _, predicted = torch.max(output.data, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                optimizer.zero_grad()

                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() / images.shape[0]
                accuracy = 100 * correct / total

                train_loss_per_batch[batch_idx] = train_loss
                accuracy_per_batch[batch_idx] = accuracy

            avg_epoch_train_loss = np.mean(train_loss_per_batch)
            avg_epoch_accuracy = np.mean(accuracy_per_batch)

            val_loss, val_accuracy, _, _, _ = self.evaluate_model(
                self.model_on_device,
                self.val_loader
            )

            if self.scheduler is not None:
                self.scheduler.step()

            write_to_tensorboard(writer, avg_epoch_train_loss, avg_epoch_accuracy, val_loss, val_accuracy, epoch)

            print(
                "Epoch: {}/{} \t Training Loss: {:.4f}, Accuracy: {:.2f}, Testing Loss: {:.4f}, Accuracy: {:.2f}".format(
                    epoch, num_epochs, train_loss, accuracy, val_loss, val_accuracy
                )
            )

            # Check early stopping conditions
            early_stopping(val_loss, self.model_on_device)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        ## Please revisit this error in names -> test/val
        stop = time.time()

        duration_s = stop - start
        writer.add_text("optimzer_parameters", str(optimizer_params_dict))
        writer.add_text("model", str(self.model_on_device))
        writer.add_text("Duration_s", str(duration_s))

        # dummy_data =  torch.randn(256, 1, 128 ,45)
        # dummy_data = dummy_data.to(self.device, dtype=torch.float)
        # writer.add_graph(model=model_on_device, input_to_model=dummy_data)

        writer.close()


    def evaluate_model(self,model_on_device, data_loader, show_cm=False):

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
            self.generate_confusion_matrix(self.classes, y_tot, y_pred_tot)

        return running_loss / num_batches, accuracy, errors, y_pred_errors, y_true_errors

    @staticmethod
    def generate_confusion_matrix(classes, y_tot, y_pred_tot):
        cm = confusion_matrix(y_tot.numpy(), y_pred_tot.numpy())
        num_classes = len(classes)
        np.set_printoptions(precision=4)

        # Coloured confusion matrix
        plt.figure(figsize=(12, 12))
        cm = confusion_matrix(y_tot.numpy(), y_pred_tot.numpy(), normalize="true")
        plt.imshow(cm, cmap=plt.cm.Blues)

        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, "{:0.3f}".format(z), ha="center", va="center")

        plt.xticks(range(num_classes))
        plt.yticks(range(num_classes))
        plt.xlabel("Prediction")
        plt.ylabel("True")

        plt.gca().set_xticklabels(classes)
        plt.gca().set_yticklabels(classes)

        plt.title("Normalized Confusion Matrix for the Data")
        plt.colorbar()
        plt.show()


def get_loaders_and_classes(data, batch_size) -> dict:
    # Define the transforms
    transform = transforms.Compose([transforms.ToTensor()])
    # Create the datasets
    train_dataset = HAVSDataset(data['x_train'], data['y_train'], transform=transform)
    val_dataset = HAVSDataset(data['x_val'], data['y_val'], transform=transform)
    test_dataset = HAVSDataset(data['x_test'], data['y_test'], transform=transform)
    classes = train_dataset.enc.classes_.tolist()

    train_idxs = np.arange(0, len(train_dataset))
    val_idxs = np.arange(0, len(val_dataset))

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idxs)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idxs)

    # Creating the data loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, sampler=train_subsampler
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, sampler=val_subsampler
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return {'train_loader':train_loader, 'val_loader':val_loader, 'test_loader':test_loader, 'classes':classes}


def write_to_tensorboard(writer, avg_epoch_train_loss, avg_epoch_accuracy, val_loss, val_accuracy, epoch):
    writer.add_scalar("Loss/Train", avg_epoch_train_loss, epoch)
    writer.add_scalar("Accuracy/Train", avg_epoch_accuracy, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

def get_info(optimizer):
    optimizer_name = optimizer.__class__.__name__
    optimizer_params_dict = {
        key: group[key]
        for group in (optimizer.param_groups)
        for key in sorted(group.keys())
        if key != "params"
    }

    return optimizer_name, optimizer_params_dict