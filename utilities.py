from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
from typing import Tuple, List, Any

def dice_metric(predicted: torch.Tensor, target: torch.Tensor) -> float:
    '''
    Calculates the Dice coefficient metric.
    
    Args:
        predicted: The predicted output from the model.
        target: The ground truth label.
        
    Returns:
        The Dice metric value (1 - DiceLoss).
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def train(model: torch.nn.Module, 
          data_in: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader], 
          loss: torch.nn.Module, 
          optim: torch.optim.Optimizer, 
          max_epochs: int, 
          model_dir: str, 
          test_interval: int = 1, 
          device: torch.device = torch.device("cuda:0")) -> None:
    '''
    Trains the model and evaluates it on the test set.
    
    Args:
        model: The neural network model.
        data_in: A tuple containing (train_loader, test_loader).
        loss: The loss function.
        optim: The optimizer.
        max_epochs: Maximum number of epochs to train.
        model_dir: Directory to save model checkpoints and metrics.
        test_interval: Interval (in epochs) to evaluate on the test set.
        device: The device to run training on (CPU or GPU).
    '''
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1

            image = batch_data["image"]
            label = batch_data["label"]
            label = label != 0
            image, label = (image.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(image)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:

            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader:

                    test_step += 1

                    test_image = test_data["image"]
                    test_label = test_data["label"]
                    test_label = test_label != 0
                    test_image, test_label = (test_image.to(device), test_label.to(device),)
                    
                    test_outputs = model(test_image)
                    
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric
                    
                
                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
                
                print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

def calculate_pixels(data: List[dict]) -> np.ndarray:
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val

def calculate_weights(val1: int, val2: int) -> torch.Tensor:
    '''
    In this function we take the number of the background and the forgroud pixels to return the `weights` 
    for the cross entropy loss values.
    '''
    count = np.array([val1, val2])
    summ = count.sum()
    weights = count/summ
    weights = 1/weights
    summ = weights.sum()
    weights = weights/summ
    return torch.tensor(weights, dtype=torch.float32)
