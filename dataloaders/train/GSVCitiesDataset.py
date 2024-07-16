import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import glob
from tqdm.notebook import tqdm

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = '/content/datasets/gsv_xs/train/'
DATAFRAME_PATH = '/content/drive/MyDrive/Project/code/datasets/gsv_xs/'


if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')


class GSVCitiesDataset(Dataset):
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.cities = [city.lower() for city in cities]

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
    
    def process_file_name(self, file_name):
        # Split the file content using '@' as delimiter
        data_tokens = file_name.split('@')

        # Extract latitude and longitude
        easting = data_tokens[1]
        northing = data_tokens[2]
        zone = data_tokens[3]
        grid_zone = data_tokens[4]
        latitude = data_tokens[5]
        longitude = data_tokens[6]

        # Extract pano ID
        pano_id = data_tokens[7]

        # Extract north degree
        north_degree = data_tokens[9]

        # Extract year and month
        year_month = data_tokens[13]
        year = year_month[:4]
        month = year_month[4:]

        # Extract place ID and city ID
        place_id = data_tokens[14]

        # Structure the extracted data as a dictionary
        img_metadata = {
            'easting': easting,
            'northing': northing,
            'zone': zone,
            'grid_zone': grid_zone,
            'latitude': latitude,
            'longitude': longitude,
            'pano_id': pano_id,
            'north_degree': north_degree,
            'year': year,
            'month': month,
            'place_id': place_id

        }
        return img_metadata
    
    def __createdataframe(self):
        column_names = ['easting', 'northing', 'zone', 'grid_zone', 'latitude', 'longitude', 'pano_id', 'north_degree', 'year', 'month', 'place_id']

        list_img_metadata = []

        # Process all records in all folders to obtain a token format of the record and save them in a list
        for city in tqdm(self.cities, 'Generating and loading the gsv_cities.csv'):
            city_path = os.path.join(self.base_path, city)
            for filename in os.listdir(city_path):
                # Process the file using your process_file function
                img_metadata = self.process_file_name(filename)
                list_img_metadata.append(img_metadata)
        df = pd.DataFrame(list_img_metadata, columns=column_names)
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]        
        
        res.set_index('place_id', inplace=True)
        res.to_csv(DATAFRAME_PATH+'gsv_cities.csv')
        return res
        
    def __getdataframes(self):
        file_csv = glob.glob('*.csv', root_dir=DATAFRAME_PATH)
        
        if not file_csv:
          df = self.__createdataframe()
        else:
          print("Loading gsv_cities.csv")
          df = pd.read_csv(DATAFRAME_PATH+file_csv[0])
          df.set_index('place_id', inplace=True)
        
        return df 
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        city = place_id.split('_')[1].lower()
        
        # get the place in form of a dataframe (each row corresponds to one image)
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + city + '/' + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(index).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)


    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')


    @staticmethod
    def get_img_name(row):
        # Construct the image name from metadata
        whole = str(row['easting']).split('.')[0].zfill(7)
        decimal = str(row['easting']).split('.')[1]
        if len(decimal)==1: decimal = decimal+'0'
        easting = whole+'.'+decimal
        whole = str(row['northing']).split('.')[0].zfill(7)
        decimal = str(row['northing']).split('.')[1]
        if len(decimal)==1: decimal = decimal+'0'
        northing = whole+'.'+decimal
        zone = str(row['zone'])
        grid_zone = str(row['grid_zone'])
        place_id = row.name
        pano_id = row['pano_id']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        north_degree = str(row['north_degree']).zfill(3)
        whole = str(row['latitude']).split('.')[0].zfill(3)
        decimal = str(row['latitude']).split('.')[1]
        if len(decimal)<5: decimal = decimal+'0'*(5-len(decimal))
        lat = whole+'.'+decimal
        whole = str(row['longitude']).split('.')[0].zfill(4)
        decimal = str(row['longitude']).split('.')[1]
        if len(decimal)<5: decimal = decimal+'0'*(5-len(decimal))
        lon = whole+'.'+decimal
        img_name = '@' + easting + '@' + northing + '@' + zone + '@' + grid_zone + '@' + lat + '@' + lon + '@' + pano_id + '@@' + \
            north_degree + '@@@@' + year + month + '@' + place_id + '@' + '.jpg'
        return img_name
