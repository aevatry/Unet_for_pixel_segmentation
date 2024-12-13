import torch
import os
from torchvision.transforms import v2
from torch.utils.data import Dataset
from PIL import Image

class SegDataset(Dataset):
    def __init__(self, img_folder:str, label_folder:str, input_size:tuple[int, int], train:bool):
        super(SegDataset, self).__init__()

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.train = train
        self.input_size = input_size

        self.all_available_files = os.listdir(img_folder)
        
        if self.train:
            # ****START CODE****
            self.augmentation = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(),
                v2.RandomGrayscale(),
            ])
        else:
            self.augmentation = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])

    def __len__(self):
        return len(self.all_available_files)
    
    def __getitem__(self, index):

        # open the right files
        filename = self.all_available_files[index]
        input_path = os.path.join(self.img_folder, filename)
        label_path = self.get_label_path(filename)

        #TODO -> implement the necessary transformations for the correct image shape as input
        
        return 

    def get_label_path(self, filename:str)->str:

        filename = filename.split(".")[0]

        full_path = os.path.abspath(os.path.join(self.label_folder, filename))
        if os.path.exists(full_path + ".png"):
            full_path += ".png"
        elif os.path.exists(full_path + ".jpg"):
            full_path += ".jpg"
        else:
            raise FileExistsError(f"the file {full_path} does not have a recognized image format")
        
        return full_path