from pathlib import Path
import argparse
import yaml
import random
from tifffile import TiffFile
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from dataset import TestDataset
from dataset import ValDataset
from dataset import LandCoverData as LCD
from torch.utils.data import DataLoader
import datetime


# def predict_as_vectors(model, dataset, steps=None):
#     """Perform a forward pass over the dataset and bincount the prediction masks to return class vectors.
#     Args:
#         model (tf.keras.Model): model
#         dataset (tf.data.Dataset): dataset to perform inference on
#         steps (int, optional): the total number of steps (batches) in the dataset, used for the progress bar
#     Returns:
#         (pandas.DataFrame): predicted class distribution vectors for the dataset
#     """
#     def bincount_along_axis(arr, minlength=None, axis=-1):
#         """Bincounts a tensor along an axis"""
#         if minlength is None:
#             minlength = tf.reduce_max(arr) + 1
#         mask = tf.equal(arr[..., None], tf.range(minlength, dtype=arr.dtype))
#         return tf.math.count_nonzero(mask, axis=axis-1 if axis < 0 else axis)
#
#     predictions = []
#     for batch in tqdm(dataset, total=steps):
#         # predict a raster for each sample in the batch
#         pred_raster = model.predict_on_batch(batch)
#
#         (batch_size, _, _, num_classes) = tuple(pred_raster.shape)
#         pred_mask = tf.argmax(pred_raster, -1) # (bs, 256, 256)
#         # bincount for each sample
#         counts = bincount_along_axis(
#             tf.reshape(pred_mask, (batch_size, -1)), minlength=num_classes, axis=-1
#         )
#         predictions.append(counts / tf.math.reduce_sum(counts, -1, keepdims=True))
#
#     predictions = tf.concat(predictions, 0)
#     return predictions.numpy()

#########"


BATCH_SIZE = 32
testDataset = TestDataset()
valDataset = ValDataset()
dataloader_test = DataLoader(testDataset, shuffle=False, batch_size=BATCH_SIZE)
dataloader_val = DataLoader(valDataset, shuffle=False, batch_size=BATCH_SIZE)

# Initialization :
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("models/unet220211119-183528_15.pth").to(device)
# device = torch.device('cpu')
print("GPU is available") if torch.cuda.is_available() else print("cpu")


def predict_as_vector(model, dataloader):
    with torch.no_grad():
        predictions = torch.empty((1,10), dtype=torch.double).to(device)
        for batch, x in tqdm(enumerate(dataloader)):

            # Initialization :
            x = x.to(device)

            # Computation :
            y_pred = model(x)
            # y_pred = torch.ones((32, 10, 256, 256)).to(device)
            (batch_size, num_classes, _, _) = tuple(y_pred.shape)
            mask = torch.argmax(y_pred, 1)
            # print(mask[0,0:10,0:10])
            # print("ms",mask.shape)
            counts = torch.sum(torch.sum(torch.eq(mask.unsqueeze(-1), torch.range(0, 9).to(device)), dim=1), dim=1).double()
            counts = counts / float(256*256)
            if batch==0:
                predictions = counts
            else:
                predictions = torch.cat((predictions, counts), dim=0)
            if batch == 150:
                break
    return predictions

predictions = predict_as_vector(model, dataloader_val).cpu()

test_files = sorted(Path("../data/dataset/").glob('test/images/*.tif'))
val_files = sorted(Path("../data/dataset/").glob('train/images/*.tif'))[0:150*BATCH_SIZE]
ids_s = pd.Series([int(f.stem) for f in val_files], name='sample_id', dtype='uint32')
df_y_pred = pd.DataFrame(
    predictions.numpy(), index=ids_s, columns=LCD.CLASSES
)
set = "val_set"
out_csv = "../results/" + f'epoch{1}_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_predicted.csv'
print(f"Saving prediction CSV to file {str(out_csv)}")
df_y_pred.to_csv(out_csv, index=True, index_label='sample_id')