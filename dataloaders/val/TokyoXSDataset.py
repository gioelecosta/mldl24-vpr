from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


DATASET_ROOT = "/content/datasets/tokyo_xs"
GT_ROOT = "/content/drive/MyDrive/Project/code/datasets/TokyoXS/"

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception(f'Please make sure the path {DATASET_ROOT} to Tokyo dataset is correct')

if not path_obj.joinpath('ref') or not path_obj.joinpath('query'):
    raise Exception(f'Please make sure the directories query and ref are situated in the directory {DATASET_ROOT}')

class TokyoXSDataset(Dataset):
    def __init__(self, input_transform = None):
        

        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT+'tokyoxs_dbImages.npy')
        
        # query images names
        self.qImages = np.load(GT_ROOT+'tokyoxs_qImages.npy')
        
        # ground truth
        self.ground_truth = np.load(GT_ROOT+'tokyoxs_gt.npy', allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+"/test/"+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)