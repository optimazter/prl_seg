import torch
from torch import nn
from rich.progress import Progress




def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):

    model = model.to(device)
    losses_train, losses_val = [], []
    
    optimizer.zero_grad(set_to_none=True)

    with Progress() as progress:

        training_task = progress.add_task("Starting training...", total = epochs)

        for epoch in range(1, epochs + 1):

            model.train()
            loss_train, loss_val = 0.0, 0.0

            for img, label in train_loader:
                img = img.to(device)
                label = label.to(device)

                outputs = model(img)
                loss = loss_fn(outputs, label)
                loss_train += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()

            with torch.no_grad():
                for img, label in val_loader:
                    img = img.to(device)
                    label = label.to(device)
                    outputs = model(img)
                    loss = loss_fn(outputs, label)
                    loss_val += loss.item()


            loss_train = loss_train / len(train_loader)
            loss_val = loss_val / len(val_loader)

            losses_train.append(loss_train)
            losses_val.append(loss_val)
            
            progress.update(training_task, advance=1, description=f"Epoch: {epoch}  |  Training loss: {loss_train}  |  Validation loss: {loss_val}")
            progress.refresh()





class DiceLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = 0


    def forward(self, prediction, label):
        assert prediction.shape == label.shape
        C = prediction.shape[0]
        assert(C == self.num_classes)

        prediction = torch.softmax(prediction, 1)

