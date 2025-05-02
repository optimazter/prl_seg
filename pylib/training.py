import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy, Metric
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import Progress
import os 
import pandas as pd
from datetime import datetime as dt
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple

import json
#from vis import plot_loss

from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from monai.transforms import (
    AsDiscrete,
    Compose
)


def train_prl_seg_unet(model, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer, 
                loss_fn, 
                epochs: int, 
                device: torch.device, 
                save_path: str = "output/monai_unet.pt",
                save_last_epoch_val: bool = False,
                save_last_epoch_val_path: str = "output/last_epoch_val.pt",
                reduce_lr_on_plateau: bool = False,
                reduce_lr_on_plateau_patience: int = 5,
                reduce_lr_on_plateau_factor: float = 0.5,
              ) -> Tuple[list, list]:

    model = model.to(device)

    losses_train, losses_val = [], []
    optimizer.zero_grad(set_to_none=True)

    dice_metric_total = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_metric_lesions = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_prls = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=reduce_lr_on_plateau_factor, patience=reduce_lr_on_plateau_patience)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        available_memory = torch.cuda.get_device_properties(device).total_memory / 1e6

    with Progress() as progress:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        epoch_task = progress.add_task(f"Epoch: 0 / {epochs}", total = epochs)
        memory_task = progress.add_task("Memory usage...", total = 100)

        for epoch in range(1, epochs + 1):

            if reduce_lr_on_plateau:
                progress.update(epoch_task, description=f"Epoch: {epoch} / {epochs} | LR {scheduler.get_last_lr()}")
             
            model.train()
            loss_train, loss_val = 0.0, 0.0
            message = f"Training epoch {epoch}..."
            train_task = progress.add_task(message, total = len(train_loader))
            
            for mag, phase, label in train_loader:
                mag = mag.to(device)
                phase = phase.to(device)
                label = label.to(device, dtype=torch.float32)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                    outputs = model(mag, phase)
                    loss = loss_fn(outputs, label)
                
                if torch.cuda.is_available():
                    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
                    progress.update(memory_task, completed=peak_memory_mb / available_memory * 100, description=f"Memory usage: {peak_memory_mb:.2f} MB")


                loss_train += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                progress.update(train_task, advance=1)
                progress.refresh()
            
            description = f"Epoch {epoch} | Training Loss: {loss_train / len(train_loader)}"
            progress.update(train_task, description=description)
            model.eval()

            val_task = progress.add_task("Validation...", total = len(val_loader))

            last_epoch_val_mag, last_epoch_val_phase, last_epoch_val_pred, last_epoch_val_label = [], [], [], []
            
            with torch.no_grad():
                for mag, phase, label in val_loader:
                    mag = mag.to(device)
                    phase = phase.to(device)
                    label = label.to(device, dtype=torch.float32)

                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                        val_outputs = model(mag, phase)
                        val_loss = loss_fn(val_outputs, label)

                    pred = torch.sigmoid(val_outputs)
                    pred = pred > 0.5

                    if save_last_epoch_val and epoch == epochs:
                        last_epoch_val_mag.append(mag.detach().cpu())
                        last_epoch_val_phase.append(phase.detach().cpu())
                        last_epoch_val_pred.append(pred.detach().cpu())
                        last_epoch_val_label.append(label.detach().cpu())


                    dice_metric_total(pred, label)
                    dice_metric_lesions(pred[:, 1], label[:, 1])
                    dice_metric_prls(pred[:, 2], label[:, 2])

                    loss_val += val_loss.item()

                    progress.update(val_task, advance=1)
                    progress.refresh()
                
                dice_metric_total_result = dice_metric_total.aggregate().item()
                dice_metric_lesions_result = dice_metric_lesions.aggregate().item()
                dice_metric_prls_result = dice_metric_prls.aggregate().item()
                
                    
            description = f"Epoch {epoch} | Validation loss: {loss_val / len(val_loader)} \n  - DiceMetric {dice_metric_total_result:.5f} \n  - DiceMetric Lesions: {dice_metric_lesions_result:.5f} \n  - DiceMetric PRLs: {dice_metric_prls_result:.5f}"
            progress.update(val_task, description=description)

            dice_metric_total.reset()
            dice_metric_lesions.reset()
            dice_metric_prls.reset()

            loss_train = loss_train / len(train_loader)
            loss_val = loss_val / len(val_loader)

            losses_train.append(loss_train)
            losses_val.append(loss_val)
            
            progress.update(epoch_task, advance=1, description=f"Epoch: {epoch} / {epochs}")
            progress.refresh()

            if reduce_lr_on_plateau:
                scheduler.step(loss_val)

        
            
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        if save_last_epoch_val:
            training_result_val = TensorDataset(torch.cat(last_epoch_val_mag), torch.cat(last_epoch_val_phase), torch.cat(last_epoch_val_pred), torch.cat(last_epoch_val_label))
            torch.save(training_result_val, save_last_epoch_val_path)
        
        return losses_train, losses_val


def train_monai_unet(model, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer, 
                loss_fn, 
                metric,
                epochs: int, 
                device: torch.device, 
                save_path: str = "output/monai_unet.pt",
                early_stopping: bool = False,
                patience: int = 5,
                min_delta: float = 1e-4,
              ):

    model = model.to(device)

    losses_train, losses_val = [], []
    
    optimizer.zero_grad(set_to_none=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        available_memory = torch.cuda.get_device_properties(device).total_memory / 1e6

    with Progress() as progress:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        epoch_task = progress.add_task(f"Epoch: 0 / {epochs}", total = epochs)
        memory_task = progress.add_task("Memory usage...", total = 100)

        for epoch in range(1, epochs + 1):
             
            model.train()
            loss_train, loss_val = 0.0, 0.0

            train_task = progress.add_task(f"Training epoch {epoch}...", total = len(train_loader))
            
            for img, label in train_loader:
                img = img.to(device)
                label = label.to(device, dtype=torch.float32)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                    outputs = model(img)
                    loss = loss_fn(outputs, label)
                
                if torch.cuda.is_available():
                    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
                    progress.update(memory_task, completed=peak_memory_mb / available_memory * 100, description=f"Memory usage: {peak_memory_mb:.2f} MB")


                loss_train += loss.item()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                progress.update(train_task, advance=1)
                progress.refresh()
            
            description = f"Epoch {epoch} | Training Loss: {loss_train / len(train_loader)}"
            progress.update(train_task, description=description)
            model.eval()

            val_task = progress.add_task("Validation...", total = len(val_loader))
            
            with torch.no_grad():
                for img, label in val_loader:
                    img = img.to(device)

                    label = label.to(device, dtype=torch.float32)

                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                        val_outputs = model(img)
                        val_loss = loss_fn(val_outputs, label)

                    pred = torch.sigmoid(val_outputs)
                    pred = pred > 0.5
                    preds = decollate_batch(pred)
                    target = decollate_batch(label)

                    metric(preds, target)

                    loss_val += val_loss.item()

                    progress.update(val_task, advance=1)
                    progress.refresh()
                
                metric_result = metric.aggregate().item()
                
                    
            description = f"Epoch {epoch} | Validation loss: {loss_val / len(val_loader)} | Validation Metric: {str(metric.__class__.__name__).split('.')[-1]} {metric_result:.5f}"
            progress.update(val_task, description=description)

            metric.reset()

            loss_train = loss_train / len(train_loader)
            loss_val = loss_val / len(val_loader)

            losses_train.append(loss_train)
            losses_val.append(loss_val)
            
            progress.update(epoch_task, advance=1, description=f"Epoch: {epoch} / {epochs}")
            progress.refresh()

            stop = True
            if early_stopping and len(losses_val) > patience:
                for i in range(patience):
                    if losses_val[-(i + 1)] - losses_val[-(i + 2)] > min_delta:
                        stop = False
                        break
                if stop:
                    print(f"Early stopping at epoch {epoch}")
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                    return losses_train, losses_val
        
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        return loss_train, loss_val




class Trainer:

    __log_file = "training_log.json"

    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                raise ValueError(f"Could not load or create directory {save_dir}!")
        

        self.log_file = f"{self.save_dir}/{Trainer.__log_file}"
        self.model: str = None
        self.timestamp: float = None
        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.loss_fn = None
        self.optimizer = None
        self.epochs: int = None
        self.batch_siz: int = None
        self.lr: float = None
        self.device: torch.device = None
        self.loss_train: list = None
        self.loss_val: list = None


    
    def __call__(self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        loss_fn, 
        optimizer, 
        epochs: int, 
        batch_size: int, 
        lr: float, 
        device: torch.device, 
        metrics: List[Metric[torch.Tensor]] = None
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.metrics = metrics
        self.loss_train, self.loss_val = self._train()
        self.timestamp = dt.now().timestamp()
        self.model_path = f"{self.save_dir}/{self.model.__class__.__name__}_{self.timestamp}.pt"
        self._save()



        
    def _save(self):
        torch.save(self.model.state_dict(), self.model_path)
        data : list = []
        if os.path.isfile(self.log_file):
            with open(self.log_file, "r") as f:
                data = json.load(f)    
        with open(self.log_file, "w") as f:
            data.append(self._to_json())
            json.dump(data, f)


    def _to_json(self):
        if (os.path.isfile(self.log_file)):
            return {
                "model": self.model.__class__.__name__,
                "timestamp": self.timestamp,
                "training_loss": self.loss_train, 
                "validation_loss": self.loss_val, 
                "loss_function": self.loss_fn.__class__.__name__, 
                "optimizer": self.optimizer.__class__.__name__, 
                "learning_rate": self.lr, 
                "batch_size": self.batch_size, 
                "epochs": self.epochs
            }
        else:
            raise ValueError("Log file does not exist, make sure the training has completed!")
    
    def _from_json(self, json):
        return Trainer(
            json["model"], 
            json["timestamp"],
            json["training_loss"], 
            json["validation_loss"], 
            json["loss_function"], 
            json["optimizer"], 
            json["learning_rate"], 
            json["batch_size"], 
            json["epochs"]
        )
            
    
    


    def _train(self):

        self.model = self.model.to(self.device)

        losses_train, losses_val = [], []
        
        self.optimizer.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats()
        available_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e6

        with Progress() as progress:

            scaler = torch.amp.GradScaler("cuda", enabled=True)

            epoch_task = progress.add_task(f"Epoch: 0 / {self.epochs}", total = self.epochs)
            memory_task = progress.add_task("Memory usage...", total = 100)

            for epoch in range(1, self.epochs + 1):
                
            
                self.model.train()
                loss_train, loss_val = 0.0, 0.0

                train_task = progress.add_task(f"Training epoch {epoch}...", total = len(self.train_loader))
                
                for img, label in self.train_loader:
                    img = img.to(self.device)
                    label = label.to(self.device, dtype=torch.float32)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                        outputs = self.model(img)
                        loss = self.loss_fn(outputs, label)
                    
                    peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6
                    progress.update(memory_task, completed=peak_memory_mb / available_memory * 100, description=f"Memory usage: {peak_memory_mb:.2f} MB")

                    loss_train += loss.item()
                    
                    if self.metrics:
                        pred = torch.sigmoid(outputs) 
                        pred = torch.flatten(pred)
                        target = torch.flatten(label)
                        for metric in self.metrics:
                            metric.update(pred, target.to(torch.long))
                

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    progress.update(train_task, advance=1)
                    progress.refresh()
                
                description = f"Epoch {epoch} | Training Loss: {loss_train / len(self.train_loader)}"

                if self.metrics:
                    description += "\nMetrics:"
                    for metric in self.metrics:
                        description += "\n{}: {:.3f}".format(str(metric.__class__.__name__), metric.compute().item())
                        metric.reset()

                progress.update(train_task, description=description)
                self.model.eval()

                val_task = progress.add_task("Validation...", total = len(self.val_loader))

                with torch.no_grad():
                    for img, label in self.val_loader:
                        img = img.to(self.device)
                        label = label.to(self.device, dtype=torch.float32)

                        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):

                            outputs = self.model(img)
                            loss = self.loss_fn(outputs, label)

                        loss_val += loss.item()
                        if self.metrics:
                            pred = torch.sigmoid(outputs)
                            pred = torch.flatten(pred)
                            target = torch.flatten(label)
                            for metric in self.metrics:
                                metric.update(pred, target.to(torch.long))

                        progress.update(val_task, advance=1)
                        progress.refresh()
                
                description = f"Epoch {epoch} | Validation Loss: {loss_val / len(self.val_loader)}"
                if self.metrics:
                    description += "\nMetrics:"
                    for metric in self.metrics:
                        description += "\n{}: {:.3f}".format(str(metric.__class__.__name__).split(".")[-1], metric.compute().item())
                        metric.reset()

                progress.update(val_task, description=description)

                loss_train = loss_train / len(self.train_loader)
                loss_val = loss_val / len(self.val_loader)

                losses_train.append(loss_train)
                losses_val.append(loss_val)
                
                progress.update(epoch_task, advance=1, description=f"Epoch: {epoch} / {self.epochs}")
                progress.refresh()
            return loss_train, loss_val


def load_latest_state_dict():
    with open(Trainer.log_file, "r") as f:
        data = json.load(f)
        latest = data[-1]
        state_dict = torch.load(latest["model_path"], weights_only=False, map_location=torch.device('cpu'))
        return state_dict





class MyDiceLoss(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.weight = weight

    def __call__(self, input, target, epsilon = 1e-5):
        target = target.unsqueeze(1)
        pred = self.sigmoid(input)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        #Epislon is added to avoid division by zero
        dice = (2. * intersection + epsilon) / (union + epsilon)
        return 1 - dice.mean()




