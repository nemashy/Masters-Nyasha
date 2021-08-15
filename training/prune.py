import numpy as np
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.utils.prune as prune


import torch.nn.functional as F
from model_training import *
# from CNN import ErnNetQAT
from utils import *
from cross_validation import *
from utils import get_pytorch_model

config = read_params('training/settings.yaml')
batch_size = config['batch_size']
num_epochs = config['number_epochs']
lr = config['lr']
transition_steps = config['transition_steps']
gamma = config['gamma_value']

# Extracting the training, validation and testing data
compressed_data_path = config['compressed_data_path']
data = decompress_data(compressed_data_path)

# Get data loaders
data_loaders_and_classes = get_loaders_and_classes(data, batch_size)
processed_data = np.load(compressed_data_path)


def prune_resnet(model,sparsity):
    """Prune resnet model"""
    # model = models.resnet18(pretrained=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 6)
    # Change the number of input channels 
    # depending on example properties
    # model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    parameters_to_prune = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
                )

    return model

def prune_ernet(model, sparsity):
    """Prune the self-named Ernet model"""
    parameters_to_prune = (
        (model.first_layer.conv1, 'weight'),
        (model.block1.conv1, 'weight'),
        (model.block1.conv2, 'weight'),
        (model.block2.conv1, 'weight'),
        (model.block2.conv2, 'weight'),
        (model.block3.conv1, 'weight'),
        (model.block3.conv2, 'weight'),
        (model.fc1, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    ) 
    return model

def measure_accuracy_ernet(model, criterion, data_loaders_and_classes,model_type):
    """Measure accuracy of a model for different sparsities"""
    for sparsity in np.linspace(0.9,0,10):
        if model_type=='ernet':
            pruned_model = prune_ernet(model, sparsity)
        elif model_type=='resnet':
            pruned_model = prune_resnet(model, sparsity)

        model_trainer = ModelTrainer(model, criterion, data_loaders_and_classes)
        model.to('cuda')
        _, accuracy, _, _, _ = model_trainer.evaluate_model(model, data_loaders_and_classes['val_loader'], False)
        yield sparsity, accuracy
        

def main():

    device = get_device()


    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 6)
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    weights_path = 'training/checkpoint.pt'
    model = get_pytorch_model(weights_path, model_ft) # ErnNet()
        # Initialise model
    # model = ErnNet()
    # model = get_pytorch_model('checkpoint.pt', model)
    # Initialising training parameters
    class_weights = get_class_weights(data['y_train'], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr)

    # Scheduling parameters
    scheduler = Scheduler(optimizer, transition_steps, gamma)
    lr_scheduler = scheduler.get_MultiStepLR()
    results = measure_accuracy_ernet(model, criterion, data_loaders_and_classes,'resnet')
    for sparsity, accuracy in results:
        print(sparsity, accuracy)

if __name__ == "__main__":
    main()