import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def get_image_path(root, basename, extension):
    return os.path.join(root, '{basename}{extension}'.format(**locals()))

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


# Standard Pascal VOC format
class VOC(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages')
        self.labels_root = os.path.join(root, 'SegmentationClass')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(get_image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(get_image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class VOCTrain(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages_train')
        self.labels_root = os.path.join(root, 'SegmentationClass_train')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(get_image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(get_image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class VOCVal(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages_val')
        self.labels_root = os.path.join(root, 'SegmentationClass_val')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(get_image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(get_image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class VOCNoLabel(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages')

        self.filenames = [image_basename(f)
                          for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        image_path = get_image_path(self.images_root, filename, '.jpg')
        with open(image_path, 'rb') as f:
            image = load_image(f).convert('RGB')

        if self.input_transform is not None:
            image = self.input_transform(image)

        return image, image_path

    def __len__(self):
        return len(self.filenames)


# For Binary Segmentation (BS)
class VOCBS(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages')
        self.labels_root = os.path.join(root, 'SegmentationClass')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        image_path = get_image_path(self.images_root, filename, '.jpg')
        with open(image_path, 'rb') as f:
            image = load_image(f).convert('RGB')
        
        label_path = get_image_path(self.labels_root, filename, '.png')
        with open(label_path, 'rb') as f:
            label = load_image(f).convert('L')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class VOCBSTrain(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages_train')
        self.labels_root = os.path.join(root, 'SegmentationClass_train')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        image_path = get_image_path(self.images_root, filename, '.jpg')
        with open(image_path, 'rb') as f:
            image = load_image(f).convert('RGB')
        
        label_path = get_image_path(self.labels_root, filename, '.png')
        with open(label_path, 'rb') as f:
            label = load_image(f).convert('L')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class VOCBSVal(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages_val')
        self.labels_root = os.path.join(root, 'SegmentationClass_val')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        image_path = get_image_path(self.images_root, filename, '.jpg')
        with open(image_path, 'rb') as f:
            image = load_image(f).convert('RGB')
        
        label_path = get_image_path(self.labels_root, filename, '.png')
        with open(label_path, 'rb') as f:
            label = load_image(f).convert('L')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)