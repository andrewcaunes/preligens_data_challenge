"""
Classes and functions to handle data
"""
from pathlib import Path
from collections import OrderedDict
import numpy as np
from tifffile import TiffFile
from torch.utils.data import Dataset
import random
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import Normalize
from torchvision.transforms import RandomRotation
import torchvision.transforms as transforms
from timeit import default_timer as timer
import tensorflow as tf
# import tensorflow_io as tfio


class LandCoverData():
    """Class to represent the S2GLC Land Cover Dataset for the challenge,
    with useful metadata and statistics.
    """
    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES = [
        'no_data',
        'clouds',
        'artificial',
        'cultivated',
        'broadleaf',
        'coniferous',
        'herbaceous',
        'natural',
        'snow',
        'water'
    ]
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES_IDX = [0, 1]

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: classes indices and RGB colors
    CLASSES_COLORPALETTE = {
        0: [0,0,0],
        1: [255,25,236],
        2: [215,25,28],
        3: [211,154,92],
        4: [33,115,55],
        5: [21,75,35],
        6: [118,209,93],
        7: [130,130,130],
        8: [255,255,255],
        9: [43,61,255]
        }
    CLASSES_COLORPALETTE = {c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()}

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [0, 20643, 60971025, 404760981, 277012377, 96473046, 333407133, 9775295, 1071, 29404605]
    )
    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356


def numpy_parse_image_mask(image_path):
    """Load an image and its segmentation mask as numpy arrays and returning a tuple
    Args:
        image_path : path to image
    Returns:
        (numpy.array[uint16], numpy.array[uint8]): the image and mask arrays
    """
    # image_path = Path(image_path)
    # get mask path from image path:
    # image should be in a images/<image_id>.tif subfolder, while the mask is at masks/<image_id>.tif
    mask_path = image_path.parent.parent/'masks'/image_path.name
    with TiffFile(image_path) as tifi, TiffFile(mask_path) as tifm:
        image = tifi.asarray()
        mask = tifm.asarray()
        # add channel dimension to mask: (256, 256, 1)
        mask = mask[..., None]
    return image, mask

def numpy_parse_image(image_path):
    """Load an image and its segmentation mask as numpy arrays and returning a tuple
    Args:
        image_path : path to image
    Returns:
        (numpy.array[uint16], numpy.array[uint8]): the image and mask arrays
    """
    # image_path = Path(image_path)
    # get mask path from image path:
    # image should be in a images/<image_id>.tif subfolder, while the mask is at masks/<image_id>.tif
    with TiffFile(image_path) as tifi:
        image = tifi.asarray()
    return image


class TrainDataset(Dataset):
    def __init__(self):
        # list of training sample files
        self.train_files = list(Path("../data/dataset/").expanduser().glob('train/images/*.tif'))
        # shuffle list of training samples files
        self.transform = ToTensor()
        self.transform2 = Compose([RandomVerticalFlip(p=0.5),
                                  RandomHorizontalFlip(p=0.5),
                                  RandomRotation(180),
                                  ToTensor()])

    def __getitem__(self, item):
        # start = timer()
        image, label = numpy_parse_image_mask(self.train_files[item]) # < 0.3s
        # end = timer()
        # print("time parse {}".format(end-start))
        channels, height, width = LandCoverData.N_CHANNELS, LandCoverData.IMG_SIZE, LandCoverData.IMG_SIZE
        image = image.astype("float32").reshape(channels, height, width) / LandCoverData.TRAIN_PIXELS_MAX # quick
        label = label.astype("int64").reshape(1, height, width) # quick
        # print(image.shape)
        # image = self.transform2(transforms.ToPILImage()(image)) # q
        image = self.transform(image) # q
        label = self.transform(label) # q
        return image, label

    def __len__(self):
        return len(self.train_files)


class TestDataset(Dataset):
    def __init__(self):
        # list of training sample files
        self.test_files = list(Path("../data/dataset/").expanduser().glob('test/images/*.tif'))

    def __getitem__(self, item):
        image = numpy_parse_image(self.test_files[item])
        channels, height, width = LandCoverData.N_CHANNELS, LandCoverData.IMG_SIZE, LandCoverData.IMG_SIZE
        image = image.astype("float32").reshape(channels, height, width) / LandCoverData.TRAIN_PIXELS_MAX
        return image

    def __len__(self):
        return len(self.test_files)

class ValDataset(Dataset):
    def __init__(self):
        # list of training sample files
        self.test_files = list(Path("../data/dataset/").expanduser().glob('train/images/*.tif'))

    def __getitem__(self, item):
        image = numpy_parse_image(self.test_files[item])
        channels, height, width = LandCoverData.N_CHANNELS, LandCoverData.IMG_SIZE, LandCoverData.IMG_SIZE
        image = image.astype("float32").reshape(channels, height, width) / LandCoverData.TRAIN_PIXELS_MAX
        return image

    def __len__(self):
        return len(self.test_files)