from torch.nn import parameter
import torch.nn.utils.prune as prune

from utility_functions import get_pytorch_model
from CNN import ErnNet

weights_path = 'checkpoint.pt'
model = ErnNet()

unpruned_model = get_pytorch_model(weights_path, model)

unpruned_model.first_layer

parameters_to_prune = (
    (unpruned_model.first_layer.conv1, 'weight'),
    (unpruned_model.block1.conv1, 'weight'),
    (unpruned_model.block1.conv2, 'weight'),
    (unpruned_model.block2.conv1, 'weight'),
    (unpruned_model.block2.conv2, 'weight'),
    (unpruned_model.block3.conv1, 'weight'),
    (unpruned_model.block3.conv2, 'weight'),
    (unpruned_model.fc1, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

print(model.state_dict().keys())