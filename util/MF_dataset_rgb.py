import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
import matplotlib.pyplot as plt

class MF_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[]):
        super(MF_dataset, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    
        
    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'rgb', 'JPG')
        label = self.read_image(name, 'labels', 'png')

        # Resize images
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2, 0, 1))

        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)

        # Modify label values to have 0, 1 (assuming there are 2 classes)
        label[label == 1] = 0  # You may need to adjust this based on your label values
        label[label == 2] = 1

        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data

class MF_dataset_ms(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[]):
        super(MF_dataset_ms, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image = np.asarray(PIL.Image.open(file_path))
        return image
    def read_t_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image = PIL.Image.open(file_path).convert("L")
        image = np.asarray(image, dtype=np.float32)
        return image
    
    def read_n_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image_ = PIL.Image.open(file_path)
        #2
        ndvi_array = np.asarray(image_, dtype=np.float32)
        # Normalize NDVI values to be within [0, 1], then scale to [0, 255]
        ndvi_normalized = (ndvi_array - np.min(ndvi_array)) / (np.max(ndvi_array) - np.min(ndvi_array))
        ndvi_scaled = (ndvi_normalized * 255).astype(np.uint8)
        
        return ndvi_scaled
        
    
        
    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_n_image(name, 'nir', 'TIF')
        label = self.read_image(name, 'labels', 'png')

        # Resize images
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32)[np.newaxis, :]
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)

        # Modify label values to have 0, 1 (assuming there are 2 classes)
        label[label == 1] = 0  # You may need to adjust this based on your label values
        label[label == 2] = 1

        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data

class MF_dataset3(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[]):
        super(MF_dataset3, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.n_data = len(self.names)

    def read_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image = np.asarray(PIL.Image.open(file_path))
        return image

    
        
    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'rgb', 'JPG')
        label = self.read_image(name, 'labels', 'png')

        # Resize images
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2, 0, 1))

        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)

        label[label == 0] = 0  # You may need to adjust this based on your label values
        label[label == 2] = 1
        label[label == 3] = 2
        # plt.imshow(label, cmap='viridis')  # viridis 컬러맵 사용
        # plt.colorbar()  # 컬러바 표시
        # plt.show()
        return torch.tensor(image), torch.tensor(label), name

    def __len__(self):
        return self.n_data