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


def unnormalize(tensor):
    """
    Reverses the normalization of a PyTorch image tensor.

    This function takes a normalized tensor and applies the inverse
    transformation to return the pixel values to the standard [0, 1] range.
    The mean and standard deviation values used for the original
    normalization are hardcoded within this function.

    Args:
        tensor (torch.Tensor): The normalized input tensor with a shape of
                               (C, H, W), where C is the number of channels.

    Returns:
        torch.Tensor: The unnormalized tensor with pixel values clamped to
                      the valid [0, 1] range.
    """
    # Define the mean and standard deviation used for the original normalization.
    mean = torch.tensor([0.485, 0.490, 0.451])
    std = torch.tensor([0.214, 0.197, 0.191])
    
    # Create a copy of the tensor to avoid modifying the original in-place.
    unnormalized_tensor = tensor.clone()
    
    # Apply the unnormalization formula to each channel: (pixel * std) + mean.
    for i, (m, s) in enumerate(zip(mean, std)):
        unnormalized_tensor[i].mul_(s).add_(m)
        
    # Clamp pixel values to the valid [0, 1] range to correct for floating-point inaccuracies.
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    
    # Return the unnormalized tensor.
    return unnormalized_tensor

def show_sample_images(dataset, class_names):
    """
    Displays a grid of sample images from the dataset.

    This function creates a plot showing one randomly selected image from each
    class and uses the provided `class_names` list for the titles.

    Args:
        dataset (Dataset): The dataset to visualize. Must have a '.classes'
                           attribute and support subset indexing.
        class_names (list of str): A list of formatted class names for the plot titles.
    """
    # Get the total number of classes from the dataset.
    num_classes = len(dataset.classes)
    
    # Validate that the number of class names matches the number of classes.
    assert len(class_names) == num_classes, "Length of class_names list must match the number of classes in the dataset."

    # Create a mapping of class index to all its image indices.
    class_to_indices = {i: [] for i in range(num_classes)}
    full_dataset_targets = dataset.subset.dataset.targets
    subset_indices = dataset.subset.indices
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)

    # Dynamically calculate the grid size for the plot.
    ncols = 7
    nrows = (num_classes + ncols - 1) // ncols  # Ceiling division for rows.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 2.2))

    # Loop through each class to display one random sample.
    for i, ax in enumerate(axes.flatten()):
        # Hide axes for any empty subplots.
        if i >= num_classes:
            ax.axis('off')
            continue

        # Set the plot title using the provided class names.
        class_name = class_names[i]

        # Pick a random image from the current class.
        random_image_idx = random.choice(class_to_indices[i])

        # Retrieve the image and label from the dataset.
        image, label = dataset[random_image_idx]

        # Un-normalize the image for proper display.
        # Assumes an 'unnormalize' function is available.
        image = unnormalize(image)

        # Prepare the image tensor for plotting.
        npimg = image.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(class_name)
        ax.axis('off')

    # Apply a tight layout and show the plot.
    plt.tight_layout()
    plt.show()
    
    
    
def display_torch_summary(summary_object, attr_names, display_names, depth):
    """
    Displays a torchinfo summary object as a styled HTML table.

    This function processes a summary object from the torchinfo library,
    formats it into a pandas DataFrame, and then renders it as a clean,
    readable HTML table within a Jupyter environment. It also displays
    key summary statistics like total parameters and memory usage below the table.

    Args:
        summary_object: The object returned by `torchinfo.summary()`.
        attr_names (list): A list of the layer attribute names to extract
                           (e.g., 'input_size', 'num_params').
        display_names (list): A list of the desired column headers for the
                              output table (e.g., 'Input Shape', 'Param #').
        depth (int, optional): The maximum depth of layers to display.
                               Defaults to infinity (showing all layers).
    """

    layer_data = []
    # Define the table column headers for the DataFrame with the new name.
    display_columns = ["Layer (type (var_name):depth-idx)"] + display_names

    for layer in summary_object.summary_list:
        # Only process layers that are within the specified depth.
        if layer.depth > depth:
            continue

        row = {}

        # Construct the hierarchical layer name with the var_name.
        indent = "&nbsp;"*4*layer.depth
        # NEW: Construct the layer name with the var_name included.
        layer_name = f"{layer.class_name} ({layer.var_name})"
        if layer.depth > 0:
            # Append depth and index for nested layers.
            layer_name = f"{layer_name}: {layer.depth}-{layer.depth_index}"

        row["Layer (type (var_name):depth-idx)"] = f"{indent}{layer_name}"

        # Iterate over both attribute and display names to populate row data.
        for attr, name in zip(attr_names, display_names):
            if attr == "num_params":
                # Mimic torchinfo's logic for displaying parameters.
                show_params = layer.is_leaf_layer or layer.depth == depth
                if show_params and layer.num_params > 0:
                    value = f"{layer.num_params:,}"
                else:
                    value = "--"
            else:
                # Fetch all other attributes directly.
                value = getattr(layer, attr, "N/A")

            row[name] = value
        layer_data.append(row)

    df = pd.DataFrame(layer_data, columns=display_columns)

    # Style the DataFrame for clean HTML presentation.
    styler = df.style.hide(axis="index")
    styler.set_table_styles([
        {"selector": "table", "props": [("width", "100%"), ("border-collapse", "collapse")]},
        {"selector": "th", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("background-color", "#4f4f4f"), ("color", "white"),
            ("border-bottom", "1px solid #ddd")
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"), ("padding", "8px"),
            ("border-bottom", "1px solid #ddd")
        ]},
    ]).set_properties(**{"white-space": "pre", "vertical-align": "top"})

    table_html = styler.to_html()

    # --- Summary Statistics ---
    total_params = f"{summary_object.total_params:,}"
    trainable_params = f"{summary_object.trainable_params:,}"
    non_trainable_params = f"{summary_object.total_params - summary_object.trainable_params:,}"
    total_mult_adds = f"{summary_object.total_mult_adds/1e9:.2f} GB"

    params_html = f"""
    <div style="margin-top: 20px; font-family: monospace; line-height: 1.6;">
        <hr><p><b>Total params:</b> {total_params}</p>
        <p><b>Trainable params:</b> {trainable_params}</p>
        <p><b>Non-trainable params:</b> {non_trainable_params}</p>
        <p><b>Total mult-adds:</b> {total_mult_adds}</p><hr>
    </div>"""

    input_size_mb = summary_object.total_input/(1024**2)
    fwd_bwd_pass_size_mb = summary_object.total_output_bytes/(1024**2)
    params_size_mb = summary_object.total_param_bytes/(1024**2)
    total_size_mb = (
        summary_object.total_input +
        summary_object.total_output_bytes +
        summary_object.total_param_bytes
    )/(1024**2)

    size_html = f"""
    <div style="font-family: monospace; line-height: 1.6;">
        <p><b>Input size (MB):</b> {input_size_mb:.2f}</p>
        <p><b>Forward/backward pass size (MB):</b> {fwd_bwd_pass_size_mb:.2f}</p>
        <p><b>Params size (MB):</b> {params_size_mb:.2f}</p>
        <p><b>Estimated Total Size (MB):</b> {total_size_mb:.2f}</p><hr>
    </div>"""

    # Combine all HTML parts and display.
    final_html = table_html + params_html + size_html
    display(HTML(final_html))


def training_loop(
    model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, scheduler=None, save_path=None,
):
   
    # Move the model to the specified computation device.
    model.to(device)

    # A dictionary to store the history of training and validation metrics.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "train_accuracy": [],
    }

    # Determine the number of classes from the dataset.
    num_classes = len(train_loader.dataset.classes)

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
        model.train()
        
        # Initialize variables to accumulate training loss for the current epoch.
        running_train_loss = 0.0
        train_samples_processed = 0

        # Iterate over the training data loader.
        for inputs, labels in train_loader:
            # Update the progress bar description for the current phase.
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Training]")
            
            # Move input data and labels to the designated device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Clear any previously calculated gradients.
            optimizer.zero_grad(set_to_none=True)
            # Forward pass: compute predicted outputs by passing inputs to the model.
            outputs = model(inputs)
            # Calculate the loss.
            loss = loss_function(outputs, labels)
            # perform a backward pass
            loss.backward()
            # Update the model weights.
            optimizer.step()
            
            # Update metrics with the current batch's predictions and labels.
            preds = outputs.argmax(dim=1)
            train_accuracy.update(preds, labels)
            
            # Update the count of processed validation samples.
            batch_size = inputs.size(0)
            train_samples_processed += batch_size
            
            # Accumulate the loss, weighted by the batch size.
            running_train_loss += loss.item() * batch_size
            
            # Calculate and display the running accuracy and loss.  
            display_acc = train_accuracy.compute().item()
            display_loss = running_train_loss / train_samples_processed

            pbar.set_postfix(
                train_acc=f"{display_acc:.4%}", 
                train_loss=f"{display_loss:.4f}",
                )
            
            # Update the progress bar for the batch.
            pbar.update(1)

        
        # Compute average accuracy, average loss for the epoch and store them in the history.
        epoch_train_acc = train_accuracy.compute().item()
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history["train_accuracy"].append(epoch_train_acc)
        history["train_loss"].append(epoch_train_loss)
       

        # --- Validation Phase ---
        # Set the model to evaluation mode.
        model.eval()
        # Initialize variables to accumulate validation loss.
        running_val_loss = 0.0
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
                outputs = model(inputs)
                # Calculate the validation loss.
                loss = loss_function(outputs, labels)
                
                # Update metrics with the current batch's predictions and labels.
                preds = outputs.argmax(dim=1)
                val_accuracy.update(preds, labels)
                
                # Update the count of processed validation samples.
                batch_size = inputs.size(0)
                val_samples_processed += batch_size
                
                # Accumulate the validation loss.
                running_val_loss += loss.item() * batch_size

                # Compute and display the current running validation accuracy and loss.
                current_acc = val_accuracy.compute().item()
                display_loss = running_val_loss / val_samples_processed
                pbar.set_postfix(
                    val_acc=f"{current_acc:.4%}",
                    val_loss=f"{display_loss:.4f}",
                )
                # Update the progress bar.
                pbar.update(1)

        # Calculate validation accuracy and average validation loss for the epoch, store them in the history.
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_accuracy.compute().item()
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        # Print the summary of the epoch's performance.
        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Acc: {epoch_train_acc:.4%}, "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
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
        torch.save(model.state_dict(), save_path)
        tqdm.write(f"-> Model saved to '{save_path}'")
                
    # Close the progress bar after the training loop is complete.
    pbar.close()
    
    return model, history



def plot_training_history(history, model_name="Custom DenseNet"):
    """Visualizes the training and validation history of a model.

    This function generates and displays two plots: one for training and
    validation loss, and another for validation accuracy. It also highlights
    the epoch where the highest validation accuracy was achieved.

    Args:
        history (dict): A dictionary containing the model's training history.
                        It must include the keys 'val_accuracy', 'val_loss',
                        and 'train_loss'.
        model_name (str, optional): The name of the model, used for plot
                                    titles and labels. Defaults to "Custom DenseNet".
    """
    # Find the index of the epoch with the highest validation accuracy.
    best_epoch_idx = np.argmax(history['val_accuracy'])
    # Get the best validation accuracy and the corresponding validation loss.
    best_val_acc = history['val_accuracy'][best_epoch_idx]
    best_val_loss = history['val_loss'][best_epoch_idx]

    # Print a summary of the model's performance at the best epoch.
    print("---------- Best Epoch Performance ----------")
    print(f"Model: {model_name}")
    print(f"Epoch: {best_epoch_idx + 1}")
    print(f"Validation Accuracy: {best_val_acc:.2%}")
    print(f"Validation Loss:     {best_val_loss:.4f}")
    print("------------------------------------------\n")

    # Set up the figure and subplots for displaying the history.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # Define colors for plot elements to ensure consistency.
    train_color = 'blue'
    val_color = 'red'
    best_epoch_color = 'green'

    # Plot training and validation loss on the first subplot.
    ax1.plot(history['train_loss'], label=f'{model_name} Train Loss', color=train_color, linestyle='-')
    ax1.plot(history['val_loss'], label=f'{model_name} Val Loss', color=val_color, linestyle='--')

    # Highlight the validation loss at the best-accuracy epoch with a marker.
    ax1.plot(best_epoch_idx, best_val_loss, marker='o', color=best_epoch_color, markersize=8, label='Loss When Best Acc Was Achieved')
    # Annotate the marker with its precise value.
    ax1.annotate(f'{best_val_loss:.4f}',
                 xy=(best_epoch_idx, best_val_loss),
                 xytext=(best_epoch_idx, best_val_loss + 0.1),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the loss subplot.
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy on the second subplot.
    ax2.plot(history['val_accuracy'], label=f'{model_name} Val Accuracy', color=val_color)

    # Highlight the best validation accuracy with a marker.
    ax2.plot(best_epoch_idx, best_val_acc, marker='o', color=best_epoch_color, markersize=8, label='Best Accuracy Achieved')
    # Annotate the marker with its value.
    ax2.annotate(f'{best_val_acc:.2%}',
                 xy=(best_epoch_idx, best_val_acc),
                 xytext=(best_epoch_idx, best_val_acc - 0.05),
                 ha='center', color=best_epoch_color,
                 arrowprops=dict(arrowstyle="->", color=best_epoch_color))

    # Set titles and labels for the accuracy subplot.
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Determine an appropriate interval for x-axis ticks for readability.
    num_epochs = len(history['train_loss'])
    if num_epochs > 10:
        x_ticks_interval = 2
    else:
        x_ticks_interval = 1

    # Generate tick locations (0-indexed) and corresponding labels (1-indexed).
    tick_locations = np.arange(0, num_epochs, x_ticks_interval)
    tick_labels = np.arange(1, num_epochs + 1, x_ticks_interval)

    # Apply the custom x-axis ticks to both subplots.
    ax1.set_xticks(ticks=tick_locations, labels=tick_labels)
    ax2.set_xticks(ticks=tick_locations, labels=tick_labels)

    # Adjust subplot parameters for a tight layout and display the plot.
    plt.tight_layout()
    plt.show()
    
    
    
def visualize_predictions(model, dataloader, class_names, device):
    """Visualizes model predictions on a sample of images from a dataset.

    This function randomly selects one image from each class in the provided
    dataloader. It then performs inference using the given model and displays
    the images in a grid. Each image is titled with its true and predicted
    labels, colored green for correct predictions and red for incorrect ones.

    Args:
        model (torch.nn.Module): The trained PyTorch model to use for inference.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the dataset to visualize.
        class_names (list of str): A list mapping class indices to their names.
        device (torch.device): The device (e.g., 'cuda', 'cpu') on which to perform inference.
    """
    # Prepare the model for inference.
    model.to(device)
    model.eval()

    # --- Create a mapping from class index to a list of sample indices for that class ---
    # Initialize a dictionary to hold indices for each class.
    class_to_indices = {i: [] for i in range(len(class_names))}
    # Access the targets and indices from the underlying dataset and subset.
    full_dataset_targets = dataloader.dataset.subset.dataset.targets
    subset_indices = dataloader.dataset.subset.indices
    # Populate the dictionary by mapping each sample's true label to its index within the subset.
    for subset_idx, full_idx in enumerate(subset_indices):
        label = full_dataset_targets[full_idx]
        class_to_indices[label].append(subset_idx)
    # ---

    # Create a grid of subplots to display the images.
    fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(18, 8))

    # Disable gradient computations for the inference phase.
    with torch.no_grad():
        # Loop through each class and its corresponding subplot axis.
        for i, ax in enumerate(axes.flatten()):
            # If there are more subplots than classes, turn off the extra ones.
            if i >= len(class_names):
                ax.axis('off')
                continue

            # Randomly select one image index from the current class.
            random_image_idx = random.choice(class_to_indices[i])
            
            # Get the image tensor and its true label from the dataset.
            image_tensor, true_label = dataloader.dataset[random_image_idx]
            
            # Prepare the image tensor for the model by adding a batch dimension and moving it to the device.
            image_batch = image_tensor.unsqueeze(0).to(device)

            # Pass the image through the model to get the output logits.
            outputs = model(image_batch)
            # Determine the predicted class index by finding the index of the maximum logit.
            _, pred = torch.max(outputs, 1)
            predicted_label = pred.item()
            
            # Set the title color to green for correct predictions and red for incorrect ones.
            is_correct = (predicted_label == true_label)
            title_color = 'green' if is_correct else 'red'
            # Set the subplot's title with the predicted and true labels.
            ax.set_title(
                f'Predicted: {class_names[predicted_label]}\n(True: {class_names[true_label]})',
                color=title_color
            )
            
            # Reverse the normalization of the image tensor for proper visualization.
            img_to_plot = unnormalize(image_tensor)
            
            # Convert the tensor to a NumPy array and adjust dimensions for displaying.
            ax.imshow(np.transpose(img_to_plot.numpy(), (1, 2, 0)))
            # Display the image and hide the axis ticks.
            ax.axis('off')

    # Adjust the layout to prevent titles from overlapping and show the plot.
    plt.tight_layout()
    plt.show() 
    
    
    
def plot_confusion_matrix(cm_np, labels):
    """Calculates and displays per-class accuracy, then plots a confusion matrix.

    This function first computes the accuracy for each individual class from the
    provided confusion matrix. It displays these scores with a progress bar, then
    uses scikit-learn's ConfusionMatrixDisplay to visualize the full matrix.

    Args:
        cm_np (numpy.ndarray): The confusion matrix to be plotted, where rows
                               represent true labels and columns represent
                               predicted labels.
        labels (list of str): A list of class names that correspond to the
                              matrix indices.
    """
    # --- Per-Class Accuracy Calculation ---
    correct_predictions = cm_np.diagonal()
    total_samples_per_class = cm_np.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.nan_to_num(correct_predictions / total_samples_per_class)
    class_accuracies = {label: acc for label, acc in zip(labels, per_class_acc)}

    # --- Display Per-Class Accuracy with a Progress Bar ---
    print("--- Per-Class Accuracy ---")
    for class_name, acc in tqdm(class_accuracies.items(), desc="Calculating Metrics"):
        print(f"{class_name:<20} | Accuracy: {acc:.2%}")
        time.sleep(0.05)
    print("-" * 40 + "\n")

    # --- Confusion Matrix Plotting ---
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_np, display_labels=labels)
    
    # Create a figure and axes object with the desired size
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Render the confusion matrix plot on the created axes
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    # Rotate the x-axis tick labels for better readability with long names.
    plt.xticks(rotation=45, ha="right")
    
    # Set the plot's title and axis labels.
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Display the finalized plot.
    plt.show()

def count_params(model):

    # 1. Trainable Parameters (requires_grad = True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Non-Trainable Parameters 
    # This includes frozen parameters AND buffers (like BN moving stats) to match TF
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    non_trainable_total = frozen_params + buffers

    # 3. Total Parameters
    total_params = trainable_params + non_trainable_total

    print(f"{'Total Parameters:':<25} {total_params:,}")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    print(f"{'Non-trainable Parameters:':<25} {non_trainable_total:,}")