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
    
    def read_t_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image = np.asarray(PIL.Image.open(file_path).convert("L"))
        return image
    
    def read_n_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image_ = PIL.Image.open(file_path)
        # # NDVI 이미지가 float 타입으로 저장되어 있다고 가정하고 처리
        # image = np.asarray(image_, dtype=np.float32)

        # # NDVI 값의 범위를 [0, 1]로 조정
        # image_normalized = (image + 1) / 2.0

        # # [0, 1] 범위를 [0, 255] 범위로 스케일링하여 uint8로 변환
        # image_scaled = (image_normalized * 255).astype(np.uint8)

        #2
        ndvi_array = np.asarray(image_, dtype=np.float32)
        # Normalize NDVI values to be within [0, 1], then scale to [0, 255]
        ndvi_normalized = (ndvi_array - np.min(ndvi_array)) / (np.max(ndvi_array) - np.min(ndvi_array))
        ndvi_scaled = (ndvi_normalized * 255).astype(np.uint8)

        return ndvi_scaled
    
        
    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'rgb', 'JPG')
        thermal = self.read_n_image(name, 'ndvi', 'TIF') #  1channel 여기 바꿔주면 됨. 
        label = self.read_image(name, 'labels', 'png')

        # Resize images
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2, 0, 1))
        thermal = np.asarray(PIL.Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32)[np.newaxis, :]
        # plt.imshow(thermal)
        # plt.colorbar()  # Show the color bar which indicates the intensity scale
        # plt.show()
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)

        # Modify label values to have 0, 1 (assuming there are 2 classes)
        label[label == 1] = 0  # You may need to adjust this based on your label values
        label[label == 2] = 1

        return torch.tensor(image), torch.tensor(thermal), torch.tensor(label), name

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
        image = np.asarray(PIL.Image.open(file_path).convert("L"))
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
        image = self.read_n_image(name, 'red', 'TIF')
        thermal = self.read_n_image(name, 'ndvi', 'TIF') #  1channel 여기 바꿔주면 됨. 
        label = self.read_image(name, 'labels', 'png')

        # Resize images
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32)[np.newaxis, :]
        thermal = np.asarray(PIL.Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32)[np.newaxis, :]
        # thermal = np.expand_dims(thermal, 1)
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)

        # Modify label values to have 0, 1 (assuming there are 2 classes)
        label[label == 1] = 0  # You may need to adjust this based on your label values
        label[label == 2] = 1

        return torch.tensor(image), torch.tensor(thermal), torch.tensor(label), name

    def __len__(self):
        return self.n_data
    

class MF_dataset_rgb2(Dataset):

    def __init__(self, data_dir, split, input_h=480, input_w=640, transform=[]):
        super(MF_dataset_rgb2, self).__init__()

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
        thermal = self.read_image(name, 'rgb', 'JPG') #  1channel 여기 바꿔주면 됨. 
        label = self.read_image(name, 'labels', 'png')

        # Resize images
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2, 0, 1))
        thermal = np.asarray(PIL.Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2, 0, 1))
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)

        # Modify label values to have 0, 1 (assuming there are 2 classes)
        label[label == 1] = 0  # You may need to adjust this based on your label values
        label[label == 2] = 1

        return torch.tensor(image), torch.tensor(thermal), torch.tensor(label), name

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
    
    def read_t_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image = np.asarray(PIL.Image.open(file_path).convert("L"))
        return image
    
    def read_n_image(self, name, folder, ext):
        file_path = os.path.join(self.data_dir, '%s/%s.%s' % (folder, name, ext))
        image_ = PIL.Image.open(file_path)
        ndvi_array = np.asarray(image_, dtype=np.float32)
        # Normalize NDVI values to be within [0, 1], then scale to [0, 255]
        ndvi_normalized = (ndvi_array - np.min(ndvi_array)) / (np.max(ndvi_array) - np.min(ndvi_array))
        ndvi_scaled = (ndvi_normalized * 255).astype(np.uint8)

        # plt.imshow(image_)  # viridis 컬러맵 사용
        # plt.colorbar()  # 컬러바 표시
        # plt.show()

        return ndvi_scaled

            
    def __getitem__(self, index):
        name = self.names[index]
        image = self.read_image(name, 'rgb', 'JPG')
        thermal = self.read_n_image(name, 'ndvi', 'TIF') #  1channel 여기 바꿔주면 됨. 
        label = self.read_image(name, 'labels', 'png')

        # Resize images
        image = np.asarray(PIL.Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2, 0, 1))
        #1안
        thermal = np.asarray(PIL.Image.fromarray(thermal).resize((self.input_w, self.input_h)), dtype=np.float32)[np.newaxis, :]
        #2안
    #     thermal_image = np.resize(thermal, (self.input_h, self.input_w))  # Update with the correct target size
    # # Convert the PIL Image to a numpy array with the desired type
    #     thermal_array = np.array(thermal_image, dtype=np.float32)[np.newaxis, :]
    #     # thermal = np.expand_dims(thermal, 1)
        label = np.asarray(PIL.Image.fromarray(label).resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST), dtype=np.int64)
        
        # plt.imshow(thermal)
        # plt.colorbar()  # Show the color bar which indicates the intensity scale
        # plt.show()
        
        label[label == 0] = 0  # You may need to adjust this based on your label values
        label[label == 2] = 1
        label[label == 3] = 2

        return torch.tensor(image), torch.tensor(thermal), torch.tensor(label), name

    def __len__(self):
        return self.n_data
