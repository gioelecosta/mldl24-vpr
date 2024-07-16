import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = "/content/datasets/sf_xs"
GT_ROOT = "/content/drive/MyDrive/Project/code/datasets/SanFranciscoXS"

class SanFranciscoXSDataset(Dataset):
    def __init__(self, which_ds='test', input_transform = None):

        self.which_ds = which_ds 

        assert which_ds.lower() in ['val', 'test']
      
        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(GT_ROOT+f'/sfxs_{which_ds}_dbImages.npy')
          
        # query images names
        self.qImages = np.load(GT_ROOT+f'/sfxs_{which_ds}_qImages.npy')
          
        # ground truth
        self.ground_truth = np.load(GT_ROOT+f'/sfxs_{which_ds}_gt.npy', allow_pickle=True)
          
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
          
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+f"/{self.which_ds}/"+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)