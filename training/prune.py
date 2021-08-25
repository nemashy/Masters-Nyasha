import numpy as np
import torch.optim as optim
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy


import torch.nn.functional as F
from model_training import *

# from CNN import ErnNetQAT
from utils import *
from cross_validation import *
from utils import get_pytorch_model

config = read_params("training/settings.yaml")
batch_size = config["batch_size"]
num_epochs = config["number_epochs"]
lr = config["lr"]
transition_steps = config["transition_steps"]
gamma = config["gamma_value"]

# Extracting the training, validation and testing data
compressed_data_path = config["compressed_data_path"]
data = decompress_data(compressed_data_path)

# Get data loaders
data_loaders_and_classes = get_loaders_and_classes(data, batch_size)
processed_data = np.load(compressed_data_path)


def prune_resnet(model, sparsity):
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
        (model.first_layer.conv1, "weight"),
        (model.block1.conv1, "weight"),
        (model.block1.conv2, "weight"),
        (model.block2.conv1, "weight"),
        (model.block2.conv2, "weight"),
        (model.block3.conv1, "weight"),
        (model.block3.conv2, "weight"),
        (model.fc1, "weight"),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    return model


def measure_accuracy_ernet(model, criterion, data_loaders_and_classes, model_type):
    """Measure accuracy of a model for different sparsities"""
    model.to("cuda")
    for sparsity in np.arange(0, 0.81, 0.01):
        model_copy = copy.deepcopy(model)
        if model_type == "ernet":
            pruned_model = prune_ernet(model_copy, sparsity)
        elif model_type == "resnet":
            pruned_model = prune_resnet(model_copy, sparsity)
        if sparsity == 0.19:
            torch.save(pruned_model, "pruned_model.pth")

        model_trainer = ModelTrainer(pruned_model, criterion, data_loaders_and_classes)

        _, accuracy, _, _, _ = model_trainer.evaluate_model(
            pruned_model, data_loaders_and_classes["val_loader"], False
        )
        print(f"Sparsity is {sparsity}, Accuracy is {accuracy}")
        yield sparsity, accuracy


def plot_results(results):

    # x1 = np.linspace(0.0, 5.0)
    # x2 = np.linspace(0.0, 2.0)

    # y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    # y2 = np.cos(2 * np.pi * x2)
    x, y = zip(*results)
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle("Results of pruning a model")

    ax1.plot(x, y, "-")
    ax1.set_xlabel("Sparsity (%)")
    ax1.set_ylabel("Validation Accuracy (%)")

    # ax2.plot(x2, y2, '.-')
    # ax2.set_xlabel('time (s)')
    # ax2.set_ylabel('Undamped')

    plt.show()


def main():

    device = get_device()

    # model_ft = models.resnet18(pretrained=False)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 6)
    # model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    weights_path = "training/model_ckpt/ernet.pt"

    # Initialise model
    model = ErnNet()

    # Initialising training parameters
    model = get_pytorch_model(weights_path, model)  # ErnNet()
    class_weights = get_class_weights(data["y_train"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr)

    # Scheduling parameters
    scheduler = Scheduler(optimizer, transition_steps, gamma)
    lr_scheduler = scheduler.get_MultiStepLR()
    results = measure_accuracy_ernet(
        model, criterion, data_loaders_and_classes, "ernet"
    )
    # for sparsity, accuracy in results:
    #     print(sparsity, accuracy)

    plot_results(results)


if __name__ == "__main__":
    main()
