from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(len(self.list_files))
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = Image.fromarray(image[:, :512, :])
        target_image = Image.fromarray(image[:, 512:, :])
        
        input_image, target_image = config.transform_both(input_image), config.transform_both(target_image)
        input_image = config.transform_input(input_image)
        target_image = config.transform_input(target_image)
        
        return np.array(input_image), np.array(target_image)