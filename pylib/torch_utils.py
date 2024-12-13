import torch
from torch import nn
import torch.nn.functional as F
from rich.progress import Progress
import os 
import pandas as pd
from datetime import datetime as dt
#from vis import plot_loss

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl



class Trainer:

    def __init__(self, log_file, save_dir):
        self.log_file = log_file
        self.save_dir = save_dir
        assert(os.path.isdir(self.save_dir))

        if (os.path.isfile(self.log_file)):
            self.df = pd.read_csv(self.log_file)
        else:
            self.df = pd.DataFrame({"model_path": [], "model": [], "training_loss": [], "Validation_loss": [], "loss_function": [], "optimizer": [], "learning_rate": [], "batch_size": [], "epochs": []})

        self.model = None
        self.model_path = None
        self.train_loader = None
        self.val_loader = None
        self.loss_fn = None
        self.optimizer = None
        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.device = None
        self.loss_train = None
        self.loss_val = None

    
    def __call__(self, model, train_loader, val_loader, loss_fn, optimizer, epochs, batch_size, lr, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.loss_train, self.loss_val = self._train()
        self.model_path = f"{self.save_dir}/{self._generate_model_name()}"

        torch.save(self.model.state_dict(), self.model_path)
        self._log_all()
        
    
    def _log_all(self):
        assert(self.model is not None and 
               self.model_path is not None and 
               self.loss_fn is not None and
               self.optimizer is not None and 
               self.epochs is not None and 
               self.device is not None and 
               self.loss_train is not None and 
               self.loss_val is not None)
    
        self.df = self.df.append({
            "model_path": None, 
            "model": self.model.__class__.__name__, 
            "training_loss": self.loss_train, 
            "validation_loss": self.loss_val, 
            "loss_function": self.loss_fn, 
            "optimizer": self.optimizer, 
            "learning_rate": self.lr, 
            "batch_size": self.batch_size, 
            "epochs": self.epochs
            }, ignore_index = True)
        
        self.df.to_csv(self.log_file, index = False)
        
    
    def _generate_model_name(self):
        return f"{self.model.__class__.__name__}_{dt.now().strftime('%m/%d/%Y, %H:%M:%S')}.pt"


    def _train(self):

        self.model = self.model.to(self.device)

        apply_activation_checkpointing(self.model)

        losses_train, losses_val = [], []
        
        self.optimizer.zero_grad(set_to_none=True)

        with Progress() as progress:

            epoch_task = progress.add_task("Epoch: 0  |  Training loss: none  |  Validation loss: none", total = self.epochs)

            for epoch in range(1, self.epochs + 1):

                self.model.train()
                loss_train, loss_val = 0.0, 0.0

                train_task = progress.add_task("Training...", total = len(self.train_loader))
                
                for img, label in self.train_loader:
                    img = img.to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(img)
                    loss = self.loss_fn(outputs, label)
                    loss_train += loss.item()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                    progress.update(train_task, advance=1)
                    progress.refresh()


                self.model.eval()

                val_task = progress.add_task("Validation...", total = len(self.val_loader))

                with torch.no_grad():
                    for img, label in self.val_loader:
                        img = img.to(self.device)
                        label = label.to(self.device)
                        outputs = self.model(img)
                        loss = self.loss_fn(outputs, label)
                        loss_val += loss.item()

                        progress.update(val_task, advance=1)
                        progress.refresh()


                loss_train = loss_train / len(self.train_loader)
                loss_val = loss_val / len(self.val_loader)

                losses_train.append(loss_train)
                losses_val.append(loss_val)
                
                progress.update(epoch_task, advance=1, description=f"Epoch: {epoch}  |  Training loss: {loss_train}  |  Validation loss: {loss_val}")
                progress.refresh()
            return loss_train, loss_val





class DiceLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes


    def forward(self, prediction, label):
        assert prediction.shape == label.shape
        C = prediction.shape[0]
        assert(C == self.num_classes)

        prediction = torch.softmax(prediction, 1)
        


class ShapeAwareCrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(x, logits, targets):

        N, C, H , W = targets.shape
        # Calculate log probabilities
        logp = F.log_softmax(logits, dim = 1)
        # Gather log probabilities with respect to target
        logp = logp.gather(1, targets.to(dtype=torch.long))

        print(logp.sum(1).shape)

        return - 1 * logp.sum(1) * F.cross_entropy(
            logits,
            targets,
        )