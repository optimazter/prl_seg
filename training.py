from pylib.torch_utils import *
from pylib.models import PRLSegUNet
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import config as c
import os



def main():
    torch.manual_seed(c.SEED)


    if not (os.path.isfile(c.PATH_TRAINING_DATASET) and os.path.isfile(c.PATH_VALIDATION_DATASET) and os.path.isfile(c.PATH_TEST_DATASET)):
        dataset_raw = torch.jit.load(c.PATH_RAW_DATASET)
        dataset_raw = list(dataset_raw.parameters())
        images = dataset_raw[0]
        labels = dataset_raw[1]

        #Convert from byte to float
        labels = labels.to(dtype = torch.float32)

        dataset = TensorDataset(images, labels)
        train_data, val_data, test_data = random_split(dataset, c.TRAIN_VAL_TEST_SPLIT)

        torch.save(train_data, c.PATH_TRAINING_DATASET)
        torch.save(val_data, c.PATH_VALIDATION_DATASET)
        torch.save(test_data, c.PATH_TEST_DATASET)
        

    else:
        train_data = torch.load(c.PATH_TRAINING_DATASET, weights_only=False) 
        val_data = torch.load(c.PATH_VALIDATION_DATASET, weights_only=False) 
        #test_data = torch.load(c.PATH_TEST_DATASET, weights_only=False) 


    train_loader = DataLoader(dataset=train_data, batch_size=c.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=c.BATCH_SIZE, shuffle=True)
    #test_loader = DataLoader(dataset=test_data)

    prlseg_unet = PRLSegUNet(n_channels=1, n_classes=2)

    optimizer = Adam(prlseg_unet.parameters())

    loss_fn = nn.CrossEntropyLoss()

    [loss for loss in train(
        prlseg_unet,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs = c.EPOCHS,
        device=c.DEVICE
    )]

    torch.save(prlseg_unet.state_dict(), c.PATH_MODEL)



if __name__ == "__name__":
    main()