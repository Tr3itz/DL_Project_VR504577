# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.vgg import VGG

# PLiNIO imports
from plinio.regularizers import DUCCIO

# Utils imports
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from timeit import default_timer as timer


def model_size(model: nn.Module) -> float:
    """Returns the size of the model in MB"""

    # Parameters storage
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # Buffers storage
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # Size in MB
    size_all_mb = (param_size + buffer_size) / 1024**2

    return size_all_mb


def train_model(model,
                data_loader: torch.utils.data.DataLoader,
                device: str,
                loss_fn: nn.Module,
                optimizer: torch.optim,
                optim_method: str = None,
                cost_strength: dict[str, float] = None,
                epochs: int = 5) -> dict:

    """
    Trains a model, returning training information:

        1. device: the target device on which the model has been trained
        2. total_time: total time required for training (in seconds)
        3. avg_time: time required for training per batch (in s/batch)
        4. avg_loss: loss per batch measured during the last epoch
        5. avg_acc: accuracy per batch measured during the last epoch

    IMPORTANT: only trains models with loss function + PLiNIO cost. Regularizers
    are not considered.
    """

    # Move the model to the target device
    model = model.to(device)

    # Measure training time
    total_time = 0
    # Measure average loss and accuracy
    avg_loss, avg_acc= 0, 0

    for epoch in range(epochs):
        print(f'********************\nEPOCH {epoch}\n')

        # Start timer
        epoch_start = timer()

        # Epoch stats
        running_loss, running_acc = 0, 0

        # Set TRAINING mode
        model.train()
        for batch, (X, y) in tqdm(enumerate(data_loader), unit='batch', total=len(data_loader)):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)
            y_probs = torch.softmax(y_logits, dim=1)
            y_preds = torch.argmax(y_probs, dim=1)

            # Loss
            loss = loss_fn(y_logits, y)
            if cost_strength is not None:
                for cost, strength in cost_strength.items():
                    loss += model.get_cost(cost)*strength

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Epoch stats update
            running_loss += loss.detach() # Prevent RAM saturation
            running_acc += accuracy_score(y.cpu(), y_preds.cpu())

            # Keep track of performance
            if batch % 500 == 499:
                print(f'Computed {batch}/{len(data_loader)} batches - Avg Loss: {(running_loss/batch):.5f} | Avg Accuracy: {((running_acc/batch)*100):.2f}%')

            # Compute average loss and accracy on the whole dataloader
            if batch == len(data_loader) - 1:
                avg_loss = running_loss/len(data_loader)
                avg_acc = running_acc/len(data_loader)

            # Free memory
            del X, y, y_logits, y_probs, y_preds
            torch.cuda.empty_cache()

        # Stop timer and update training time
        epoch_end = timer()
        time_elapsed = epoch_end - epoch_start
        total_time += time_elapsed

        print(f'\nEnd of EPOCH {epoch} - Avg Loss: {(avg_loss):.5f} | Avg Accuracy: {(avg_acc*100):.2f}%')
        print(f'Training time: {time_elapsed:.3f} seconds.')

        # Keep track of the optimization
        if optim_method == 'PIT' or optim_method == 'SuperNet':
            print(f"\n#Parameters after EPOCH {epoch}: {int(model.get_cost(cost).item())}")
        elif optim_method == 'MPS':
            print(f"\nStorage required for weights: {(model.get_cost(cost).item()/(8*1024*1024)):.2f} MB - {int(model.get_cost(cost).item())} bits")

        print('********************\n')

        # Free memory
        torch.cuda.empty_cache()

    # Collect results
    results = {
        'device': device,
        'total_time': total_time,
        'avg_time': total_time/epochs,
        'avg_loss': avg_loss,
        'avg_acc': avg_acc
    }

    # Move the model back to the CPU and free GPU space
    model = model.cpu() if torch.cuda.is_available() else model
    torch.cuda.empty_cache()

    return results


def train_model_duccio(model,
                       data_loader: torch.utils.data.DataLoader,
                       device: str,
                       loss_fn: nn.Module,
                       optimizer: torch.optim,
                       cost_names: list[str],
                       regularizer: DUCCIO,
                       epochs: int = 5) -> dict:

    """
    Trains a model using DUCCIO regularizer, returning training information:

        1. device: the target device on which the model has been trained
        2. total_time: total time required for training (in seconds)
        3. avg_time: time required for training per batch (in s/batch)
        4. avg_loss: loss per batch measured during the last epoch
        5. avg_acc: accuracy per batch measured during the last epoch
    """

    # Move the model to the target device
    model = model.to(device)

    # Measure training time
    total_time = 0
    # Measure average loss and accuracy
    avg_loss, avg_acc= 0, 0

    for epoch in range(epochs):
        print(f'********************\nEPOCH {epoch}\n')

        # Start timer
        epoch_start = timer()

        # Epoch stats
        running_loss, running_acc = 0, 0

        # Set TRAINING mode
        model.train()
        for batch, (X, y) in tqdm(enumerate(data_loader), unit='batch', total=len(data_loader)):
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logits = model(X)
            y_probs = torch.softmax(y_logits, dim=1)
            y_preds = torch.argmax(y_probs, dim=1)

            # Loss
            loss = loss_fn(y_logits, y) + regularizer(model, epoch, epochs)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Epoch stats update
            running_loss += loss.detach() # Prevent RAM saturation
            running_acc += accuracy_score(y.cpu(), y_preds.cpu())

            # Keep track of performance
            if batch % 500 == 499:
                print(f'Computed {batch}/{len(data_loader)} batches - Avg Loss: {(running_loss/batch):.5f} | Avg Accuracy: {((running_acc/batch)*100):.2f}%')

            # Compute average loss and accracy on the whole dataloader
            if batch == len(data_loader) - 1:
                avg_loss = running_loss/len(data_loader)
                avg_acc = running_acc/len(data_loader)

            # Free memory
            del X, y, y_logits, y_probs, y_preds
            torch.cuda.empty_cache()

        # Stop timer and update training time
        epoch_end = timer()
        time_elapsed = epoch_end - epoch_start
        total_time += time_elapsed

        print(f'\nEnd of EPOCH {epoch} - Avg Loss: {(avg_loss):.5f} | Avg Accuracy: {(avg_acc*100):.2f}%')
        print(f'Training time: {time_elapsed:.3f} seconds.\n')

        # Keep track of the optimization
        for cost in cost_names:
            print(f'{cost} after epoch EPOCH {epoch}: {int(model.get_cost(cost).item())}')

        print('********************\n')

        # Free memory
        torch.cuda.empty_cache()

    # Collect results
    results = {
        'device': device,
        'total_time': total_time,
        'avg_time': total_time/epochs,
        'avg_loss': avg_loss,
        'avg_acc': avg_acc
    }

    # Move the model back to the CPU and free GPU space
    model = model.cpu() if torch.cuda.is_available() else model
    torch.cuda.empty_cache()

    return results


def evaluate_model(model: nn.Module, device: str, loss_fn: nn.Module, data_loader: DataLoader) -> dict:
    """
    Evaluates a model, returning the following information:

        1. device: the target device on which the model has been trained
        2. total_inference_time: total time required for evaluation (in seconds)
        3. avg_inference_time: time required for evaluation per batch (in ms/batch)
        4. min_inference_time: minimum recorded inference time (in milliseconds)
        5. avg_loss: loss per batch
        6. avg_acc: accuracy per batch
    """

    # Move the model to the target device
    model = model.to(device)

    # Measure inference time
    total_inference_time = 0
    min_inference_time = None

    # Measure average loss and accuracy
    avg_loss, avg_acc = 0, 0

    # Put the model in evaluation mode
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Move the batch to the target device
            X, y = X.to(device), y.to(device)

            # Start inference
            inference_start = timer()

            # Forward pass
            y_logits = model(X)
            y_probs = torch.softmax(y_logits, dim=1)
            y_preds = torch.argmax(y_probs, dim=1)

            # End inference
            inference_end = timer()

            # Compute stattistics
            loss = loss_fn(y_logits, y)
            acc = accuracy_score(y_true=y.cpu(), y_pred=y_preds.cpu())

            # Update statistics
            avg_loss += loss.detach() # Prevent RAM saturation
            avg_acc += acc
            time_elapsed = inference_end - inference_start
            min_inference_time = time_elapsed if min_inference_time is None else min(min_inference_time, time_elapsed)                
            total_inference_time += time_elapsed

            # Free memory
            del X, y, y_logits, y_preds, y_probs
            torch.cuda.empty_cache()

        # Compute average loss and accuracy
        avg_loss /= len(data_loader)
        avg_acc /= len(data_loader)

        # Compute average inference time
        avg_inference_time = total_inference_time / len(data_loader)

    # Collect results
    inference_results = {
        'device': device,
        'total_inference_time': total_inference_time,
        'avg_inference_time': avg_inference_time*1000,
        'min_inference_time': min_inference_time*1000,
        'avg_loss': avg_loss,
        'avg_acc': avg_acc
    }

    # Move the model back to CPU and free GPU space
    model = model.cpu() if torch.cuda.is_available() else model
    torch.cuda.empty_cache()

    return inference_results