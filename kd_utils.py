import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from IPython.display import display, HTML
from sklearn.metrics import ConfusionMatrixDisplay
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
)
from torchvision import datasets
from tqdm.auto import tqdm

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """
    student_logits: output of student model, shape [B, C]
    teacher_logits: output of teacher model, shape [B, C]
    labels: ground truth labels, shape [B]
    temperature: softening factor
    alpha: weight for hard-label loss
    """
    # Hard loss with true labels
    hard_loss = F.cross_entropy(student_logits, labels)

    # Soft loss with teacher predictions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Multiply by T^2 as in standard KD
    loss = alpha * hard_loss + (1.0 - alpha) * (temperature ** 2) * soft_loss
    return loss, hard_loss, soft_loss
        
        
def student_training_loop(
    teacher, student, train_loader, val_loader, optimizer,temperature, alpha, num_epochs, device, scheduler=None, save_path=None,
):

    # Move the models to the specified computation device.
    teacher.to(device)
    student.to(device)
    
    # Ensure the teacher model is in evaluation mode and its parameters are not updated during training.
    teacher.eval()
    
    # A dictionary to store the history of training and validation metrics.
    history = {
        "train_accuracy": [],
        "train_hard_loss": [],
        "train_soft_loss": [],
        "train_distill_loss": [],
        "val_accuracy": [],
        "val_hard_loss": [],
        "val_soft_loss": [],
        "val_distill_loss": [],
    }

    # Determine the number of classes from the dataset.
    num_classes = len(train_loader.dataset.classes)
    
    # Initialize torchmetrics for calculating accuracy
    train_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)


    # Set up a single progress bar for the entire training process.
    total_steps = (len(train_loader) + len(val_loader)) * num_epochs
    pbar = tqdm(total=total_steps, desc="Overall Progress")

    # Begin the main training loop over the specified number of epochs.
    for epoch in range(num_epochs):
        # --- Training Phase ---
        
        # Reset metric calculators for the new training epoch.
        train_accuracy.reset()
        
        # Set the model to training mode.
        student.train()
        
        # Initialize variables to accumulate training loss for the current epoch.
        running_train_hard_loss = 0.0
        running_train_soft_loss = 0.0
        running_train_distill_loss = 0.0
        train_samples_processed = 0

        # Iterate over the training data loader.
        for inputs, labels in train_loader:
            # Update the progress bar description for the current phase.
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Training]")
            
            # Move input data and labels to the designated device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear any previously calculated gradients.
            optimizer.zero_grad(set_to_none=True)
            
            # teacher and student forward pass
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)

            distill_loss, hard_loss, soft_loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=temperature,
                alpha=alpha
            )
            
            # perform a backward pass
            distill_loss.backward()
            # Update the model weights.
            optimizer.step()
            
            # Update metrics with the current batch's predictions and labels.
            preds = student_logits.argmax(dim=1)
            train_accuracy.update(preds, labels)
            
            # Update the count of processed samples.
            batch_size = inputs.size(0)
            train_samples_processed += batch_size
            
            # Accumulate the loss
            running_train_hard_loss += hard_loss.item() * batch_size
            running_train_soft_loss += soft_loss.item() * batch_size
            running_train_distill_loss += distill_loss.item() * batch_size
            
            # Compute and display the current running training accuracy and loss.
            display_acc = train_accuracy.compute().item()
            display_distill_loss = running_train_distill_loss / train_samples_processed
            pbar.set_postfix(
                acc=f"{display_acc:.4%}",
                distill_loss=f"{display_distill_loss:.4f}",
                )
            
            # Update the progress bar for the batch.
            pbar.update(1)

        # calculate the average training accuracy and loss for the epoch, storing them in the history dictionary.
        epoch_train_acc = train_accuracy.compute().item()
        epoch_train_hard_loss = running_train_hard_loss / len(train_loader.dataset)
        epoch_train_soft_loss = running_train_soft_loss / len(train_loader.dataset)
        epoch_train_distill_loss = running_train_distill_loss / len(train_loader.dataset)
        
        history["train_accuracy"].append(epoch_train_acc)
        history["train_hard_loss"].append(epoch_train_hard_loss)
        history["train_soft_loss"].append(epoch_train_soft_loss)
        history["train_distill_loss"].append(epoch_train_distill_loss)

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        student.eval()
        
        # Initialize variables to accumulate validation loss.
        running_val_acc = 0.0
        running_val_hard_loss = 0.0
        running_val_soft_loss = 0.0
        running_val_distill_loss = 0.0
        val_samples_processed = 0
        
        # Reset metric calculators for the new validation epoch.
        val_accuracy.reset()

        # Disable gradient calculations for the validation phase.
        with torch.no_grad():
            # Iterate over the validation data loader.
            for inputs, labels in val_loader:
                # Update the progress bar description for the validation phase.
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Validation]")
                
                # Move input data and labels to the designated device.
                inputs, labels = inputs.to(device), labels.to(device)
               
                # Compute model outputs.
                student_logits = student(inputs)
                teacher_logits = teacher(inputs)
                
                # Update metrics with the current batch's predictions and labels
                preds = student_logits.argmax(dim=1)
                val_accuracy.update(preds, labels)
                
                # Update the count of processed validation samples.                
                batch_size = inputs.size(0)
                val_samples_processed += batch_size
                
                # Calculate the validation loss.
                loss, hard_loss, soft_loss = distillation_loss(student_logits, teacher_logits, labels)
        
                # Accumulate the loss
                running_val_hard_loss += hard_loss.item() * batch_size
                running_val_soft_loss += soft_loss.item() * batch_size
                running_val_distill_loss += loss.item() * batch_size
                
                # Compute and display the current running validation accuracy and loss.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_distill_loss / val_samples_processed
                pbar.set_postfix(
                    val_acc=f"{current_acc:.4%}",
                    distill_loss=f"{display_loss:.4f}",
                )
                # Update the progress bar.
                pbar.update(1)

        # Calculate the average validation loss  and accuracy for the epoch.
        epoch_val_hard_loss = running_val_hard_loss / len(val_loader.dataset)
        epoch_val_soft_loss = running_val_soft_loss / len(val_loader.dataset)
        epoch_val_distill_loss = running_val_distill_loss / len(val_loader.dataset)
        epoch_val_acc = val_accuracy.compute().item()
        
        # Store the epoch's validation loss and accuracy in the history.
        history["val_hard_loss"].append(epoch_val_hard_loss)
        history["val_soft_loss"].append(epoch_val_soft_loss)
        history["val_distill_loss"].append(epoch_val_distill_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print the summary of the epoch's performance.
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Acc: {epoch_train_acc:.4%}, "
            f"Train Hard Loss: {epoch_train_hard_loss:.4f}, "
            f"Train Soft Loss: {epoch_train_soft_loss:.4f}, "
            f"Train Distill Loss: {epoch_train_distill_loss:.4f} | "
            f"Val Hard Loss: {epoch_val_hard_loss:.4f}, "
            f"Val Soft Loss: {epoch_val_soft_loss:.4f}, "
            f"Val Distill Loss: {epoch_val_distill_loss:.4f}, "
            f"Val Acc: {epoch_val_acc:.4%}"
        )

        # --- SCHEDULER ---
        # Adjust the learning rate based on the scheduler's logic, if one is provided.
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_acc)
            else:
                scheduler.step()

    # Save the model's state dictionary if a save path is specified.
    if save_path:
        torch.save(student.state_dict(), save_path)
        tqdm.write(f"-> Model saved to '{save_path}'")
        
    # Close the progress bar after the training loop is complete.
    pbar.close()
    
    return student, history

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Evaluation Accuracy: {acc:.4%}")
    return acc



