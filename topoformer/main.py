# -----------------------------------------------------------------------------
# Copyright (C) 2024, Electronics and Telecommunications Research Institute (ETRI)
# All rights reserved.
#
# This code is a simple proof of concept based on the description in the paper:
# Gagrani et al., "Neural Topological Ordering for Computation Graphs", NeurIPS 2022.
#
# We refer to the 'run.py' file from Microsoft CoderBERT and its variants to implement 'main.py' and run.
#
# @Author: Youngmok Ha
#
# Date: September 27, 2024
#
# Required definitions to run this source code:
#  - loss function 
#  - dataloader
#  - method named 'get_features_masks_labels' to obtain features, masks, and labels from data.
#
# Assumption on the dimensionality:
#  - the shape of (node) features: (batch_size, num_nodes, num_features * num_graphs)
#  - the shape of masks:           (batch_size, num_nodes, num_nodes * num_graphs)
#  - the shape of labels:          (batch_size, num_nodes, num_features * num_graphs)
# -----------------------------------------------------------------------------

import os
import logging
import argparse
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np

from topoformer import Topoformer # Importing the model building function

NUM_GRAPHS = 7  # fixed for the Topoformer model

# Setting up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to initialize the model's weights
def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        # Kaiming initialization due to the GELU based networks
        nn.init.kaiming_uniform_(model.weight.data)  

# Inference function that computes predictions and calculates the loss
def inference(model, loss_func, features, masks, labels):
    predicts = model(features, masks)   # Forward pass through the model

    # Reshape predictions and labels for loss calculation
    #   predicts = predicts.contiguous().view(-1, predicts.shape[-1])
    #  or
    #   predicts = predicts.contiguous().view(-1)
    #  or 
    #   another view of predicts
    #	labels = labels.contiguous().view(-1)

    # Compute loss function
    loss = loss_func(predicts, labels)

    return loss

# Test function to evaluate the model on the valid and test data
def test(model, loss_func, test_data_iter, data_device, multi_device):
    model.eval()  # Set the model to evaluation mode
    total_test_loss = 0  # To accumulate the total loss
    with torch.no_grad():   # Disable gradient calculation for efficiency
        for x, y in test_data_iter:
            x = x.to(data_device)   
            y = y.to(data_device)   

            # Extract features, masks, and labels
            #features, masks, labels = get_features_masks_labels(x, y) 
            loss = inference(model, loss_func, features, masks, labels)  # Compute the loss for test data
            if multi_device is True:
                loss = loss.mean()  # Average the loss if using multiple devices
            total_test_loss += loss.item()  # Accumulate the loss

    test_loss = total_test_loss / len(test_data_iter)    # Average the test loss
    return test_loss

def get_device_info(device = torch.device("cpu"),
                    data_device = torch.device("cpu"),
                    num_gpus = 0,
                    multi_device = False):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        data_device = torch.device("cuda:0")
        num_gpus = torch.cuda.device_count()
        multi_device = True if num_gpus > 1 else False
        torch.cuda.empty_cache()  # Clear the GPU cache

    return device, data_device, num_gpus, multi_device

# Set random seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)  # Set seed for Python's hash-based operations
    np.random.seed(seed)
    torch.manual_seed(seed) # Set seed for PyTorch
    torch.cuda.manual_seed(seed)    # Set seed for CUDA (GPU operations)
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior for cuDNN

# Main function to handle argument parsing and model training/testing
def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--num_epochs", default=326, type=int, help="The number of training epochs to perform")

    arg_parser.add_argument("--batch_size", default=8, type=int,
                            help="Batch size per device for training and evaluation")
    arg_parser.add_argument("--num_workers", default=8, type=int, help="The number of workers for dataloader")

    arg_parser.add_argument("--learning_rate", default=1e-4, type=float,
                            help="The initial learning rate for Adam Adam optimizer")
    arg_parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay if we apply")
    arg_parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon parameter for Adam optimizer")
    arg_parser.add_argument("--max_grad_norm", default=1e-1, type=float, help="The maximum value of gradient norm")

    arg_parser.add_argument("--warmup_steps", default=15, type=int, help="Linear warmup over warmup_steps")
    arg_parser.add_argument("--schedule_factor", default=0.1, type=float, help="Factor by which the learning rate will be reduced. new_lr = lr * factor")
    arg_parser.add_argument("--schedule_patience", default=10, type=int, help="The number of allowed epochs with no improvement after which the learning rate will be reduced")

    arg_parser.add_argument("--norm_epsilon", default=1e-8, type=float, help="Epsilon parameter for normalization")
    arg_parser.add_argument("--dropout_rate", default=0.1, type=float, help="The rate for dropout weight")

    arg_parser.add_argument("--num_layers", default=8, type=int,
                            help="The number of block layers")
    arg_parser.add_argument("--num_heads", default=10, type=int,
                            help="The number of attention heads")
    arg_parser.add_argument("--num_features", default=256, type=int,
                            help="")
    arg_parser.add_argument("--dim_embed", default=256, type=int,
                            help="The dimension for embedding node feature")
    arg_parser.add_argument("--dim_qkv", default=64*10, type=int,
                            help="The dimension of query, key, and value of attention mechanism")
    arg_parser.add_argument("--dim_hidden", default=256, type=int,
                            help="The dimension of hidden layers in fully-connected neural networks")

    arg_parser.add_argument("--output_dir", default="./checkpoint", type=str,
                            help="The output directory where the model predictions and checkpoints will be saved")

    arg_parser.add_argument("--seed", default=0, type=int,
                            help="Random seed")

    # Parse arguments and log the configuration
    args = arg_parser.parse_args()
    logger.info(args)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory if it does not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # Get device and multi-device information
    device, data_device, _, multi_device = get_device_info()

    # Get the data iterators for training, evaluation, and testing
    train_data_iter, eval_data_iter, test_data_iter = dataloader(batch_size=args.batch_size,
                                                                 num_workers=args.num_workers)

    # Build the model with the specified configuration
    model = Topoformer(dim_input=args.num_features,
                       dim_embed=args.dim_embed,
                       num_layers=args.num_layers,
                       num_blocks=NUM_GRAPHS,
                       dim_qkv=args.dim_qkv,
                       num_heads=args.num_heads,
                       dim_hidden=args.dim_hidden,
                       dropout_rate=args.dropout_rate,
                       norm_epsilon=args.norm_epsilon)

    model.apply(initialize_weights) # Initialize model weights
    model.to(device)    # Move model to the device
    if multi_device is True:
        model = nn.DataParallel(model)  # Use DataParallel if multiple devices are available

    assert(next(model.parameters()).device == data_device)  # Ensure the device assignment

    # Define the optimizer and learning rate scheduler
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=args.schedule_factor, patience=args.schedule_patience)

    # Define the loss function
    #loss_func = nn.xxxx()

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0  # To accumulate the total loss
        for x, y in train_data_iter:
            optimizer.zero_grad()   # Zero the gradients

            x = x.to(data_device)             
            y = y.to(data_device) 

            # Extract features, masks, and labels
            #features, masks, labels = get_features_masks_labels(x, y)   

            loss = inference(model, loss_func, features, masks, labels)   # Compute the loss
            if multi_device is True:
                loss = loss.mean()    # Average the loss if using multiple devices
            total_train_loss += loss.item()   # Accumulate the loss

            # Backpropagation and gradient clipping
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()    # Update the model parameters

        train_loss = total_train_loss / len(train_data_iter)    # Average the training loss

        # Evaluate the model after each epoch
        eval_loss = test(model, loss_func, eval_data_iter, data_device, multi_device)

        # Adjust the learning rate after the warmup phase
        if epoch > args.warmup_steps:
            scheduler.step(eval_loss)
        logging.info(f"epoch: {epoch}, train_loss: {train_loss:.5f}, eval_loss: {eval_loss:.5f}")

    # Final test after training
    test_loss = test(model, loss_func, test_data_iter, data_device, multi_device)
    logging.info(f"test_loss: {test_loss:.5f}")

# Entry point to run the main function
if __name__ == "__main__":
    main()
