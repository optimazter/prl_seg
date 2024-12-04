from pylib.torch_utils import *
from pylib.UNet import PRLSegUNet
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import config as c


def main():

    torch.manual_seed(c.SEED)

    prlseg_unet = PRLSegUNet()
    
    train_data = torch.load(c.PATH_TRAINING_DATASET)
    val_data = torch.load(c.PATH_VALIDATION_DATASET)
    test_data = torch.load(c.PATH_TEST_DATASET)

    train_loader = DataLoader(dataset=train_data)
    val_loader = DataLoader(dataset=val_data)
    #test_loader = DataLoader(dataset=test_data)

    optimizer = Adam(prlseg_unet.parameters())

    loss_fn = 

    train(
        prlseg_unet,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ecpochs = c.EPOCHS,
        device=c.DEVICE
    )

    torch.save(prlseg_unet.state_dict(), c.PATH_MODEL)




if __name__ == "__main__":
    main()