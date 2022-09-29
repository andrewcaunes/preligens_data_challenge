from torchsummary import summary
from model import Block
from model import UNet2

import torch
from dice_score import dice_loss
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from timeit import default_timer as timer
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
from tifffile import TiffFile
import dataset as ds
import random
import numpy as np


## Speeding up
# num workers = 4*gpu_nb
# scheduler




torch.cuda.empty_cache()

if __name__ == '__main__':

    BATCH_SIZE = 10
    # Load data
    trainDataset = ds.TrainDataset()
    dataloader_train = DataLoader(trainDataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Initialize logging on tensorboard
    logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=logdir)

    # Initialization :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("GPU is available") if torch.cuda.is_available() else print("cpu")
    activation = torch.nn.Softmax()
    # model = UNet2(4, 10, bilinear=False).to(device) # 4 channels to 10 classes
    model = Block(4).to(device) # 4 channels to 10 classes
    summary(model, (4,256,256))
    # optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    learning_rate = 1e-5
    # optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    # scheduler = StepLR(optim, step_size=5000, gamma=0.5)

    # scheduler and AMP
    # scheduler = ReduceLROnPlateau(optim, 'max', patience=2)
    # cyclic_scheduler = CyclicLR(optim, base_lr=1e-4, max_lr=1e-3, cycle_momentum=False, )
    grad_scaler = torch.cuda.amp.GradScaler()

    class_weight = (1 / ds.LandCoverData.TRAIN_CLASS_COUNTS[2:]) * ds.LandCoverData.TRAIN_CLASS_COUNTS[2:].sum() / (ds.LandCoverData.N_CLASSES - 2)
    class_weight = np.concatenate((np.zeros((2,)), class_weight))
    print(class_weight)
    print(class_weight.shape)
    class_weight = torch.FloatTensor(class_weight).cuda()
    loss_fn = CrossEntropyLoss(weight=class_weight)
    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #try multiple convolutions
    torch.backends.cudnn.benchmark = True

    # Training and testing loops :
    EPOCHS = 10000

    save_step = 1
    train = True
    if train:
        start = timer()
        lr = 0
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            for batch, (x, y) in tqdm(enumerate(dataloader_train)):
                # print("time loading : {}".format(timer()-start2))
                # start = timer()
                # Initialization :
                x = x.view(-1, 4, 256, 256).to(device)
                y = y.view(-1, 1, 256, 256).to(device)
                # print("x",x[0,0,0:6,0:6])
                # print("y",y[0,0,0:6,0:6])
                # print(torch.min(y))
                # print(torch.max(y))

                # Computation :
                with torch.cuda.amp.autocast():
                    y_pred = model(x)
                    # print("yp s",y_pred.shape)
                    # print("y s",y.shape)
                    loss = loss_fn(y_pred, y.squeeze(1))
                # set params to none (instead of zero_grad)
                for group in optim.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.detach_()
                optim.zero_grad()
                # optim.zero_grad()
                loss.backward()
                optim.step()
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optim)
                # grad_scaler.update()
                # cyclic_scheduler.step()
                # Tensorboard :
                if batch % 10 == 0:
                    with torch.no_grad():
                        end = timer() - start
                        num_acc = (torch.argmax(y_pred, 1) == y.squeeze(1)).float().sum().item()
                        denum_acc = float(len(y.view(-1)))
                        accuracy_train = num_acc / denum_acc
                        writer.add_scalar("Accuracy/train", accuracy_train, int(end))
                        writer.add_scalar("Loss/train", loss, int(end))
            # test_loop(dataloader_test, model, optim, loss_fn, starting_time=start, freq=1)
            # scheduler.step()
            # if scheduler.get_lr() != lr:
            #     lr = scheduler.get_lr()
            #     print("New learning rate : ", lr)
            if epoch % save_step == 0:
                savepath = Path("models/unet2"+name+"_{}.pth".format(epoch))
                torch.save(model, savepath)
            scheduler.step()

    writer.close()

