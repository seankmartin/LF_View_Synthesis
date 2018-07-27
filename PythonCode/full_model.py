"""Contains the full model, including
Model
Optimizer
LR scheduler
Criterion
"""
import torch.nn as nn
import torch.optim as optim

from model_3d import C3D
from torch.optim.lr_scheduler import ReduceLROnPlateau

def setup_model(args):
    """Returns a tuple of the model, criterion, optimizer and lr_scheduler"""
    print("Building model")
    model = C3D(inchannels=2, outchannels=64)
    criterion = nn.MSELoss(size_average=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        'min', factor=args.lr_factor,
        patience=3, threshold=1e-3,
        threshold_mode='rel', verbose=True)

    return (model, criterion, optimizer, lr_scheduler)