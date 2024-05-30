import numpy as np 
import pandas as pd 
from scipy.signal import resample 
import random
import torch 
from torch.utils.data import Dataset, DataLoader 
from config import cfg
from sklearn.model_selection import KFold

np.random.seed(42)

class ds(Dataset):
  def __init__(self, x, y=None, transforms=None):
    super().__init__()

    self.X = x
    self.Y = y
    self.transforms = transforms

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self,idx):
    x = self.X.iloc[idx,:]
    
    if self.transforms is not None:
        x = self.transforms(x)

    if self.Y is not None:
      return torch.Tensor(x).view(1,-1).float(), torch.Tensor([self.Y.iloc[idx]]).float().squeeze()

    return torch.Tensor(x).float()
    

def stretch(x):
    l = int(187 * (1 + (random.random() - 0.5) / 3))
    y = resample(x, l)
    if l < 187:
        y_ = np.zeros(shape=(187,))
        y_[:l] = y
    else:
        y_ = y[:187]
    return y_

def amplify(x):
    alpha = (random.random() - 0.5)
    factor = -alpha * x + (1 + alpha)
    return x * factor

class Stretch:
    def __init__(self):
        pass

    def __call__(self, x):
        return stretch(x)

    def __repr__(self):
        return 'Stretch'

class Amplify:
    def __init__(self):
        pass

    def __call__(self, x):
        return amplify(x)

    def __repr__(self):
        return 'Amplify'

class Augment:
    def __init__(self, augmentation_list, return_prints=False):
        self.augmentation_list = augmentation_list
        self.return_prints = return_prints

    def __call__(self, x):
        augmentations_performed = ''
        for augmentation in self.augmentation_list:
            if np.random.binomial(1, 0.5) == 1:
                x = augmentation(x)
                augmentations_performed += f'{augmentation} '
        if not self.return_prints:
            return x
        return x, augmentations_performed

def load_data_from_csv(train_csv, test_csv):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    return df_train, df_test

def create_data_loaders(train_csv, test_csv, batch_size, augmentations=None):
    df_train, df_test = load_data_from_csv(train_csv, test_csv)
    train_set = ds(df_train.iloc[:, :-1], df_train.iloc[:, -1], transforms=augmentations)
    test_set = ds(df_test.iloc[:, :-1], df_test.iloc[:, -1])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size * 4 ,shuffle= True)
    return train_loader, test_loader, df_train, df_test

def data_processing():
    train_csv = '/Users/munzur/Desktop/FEDERATED_ECG/Data/mitbih_train.csv'
    test_csv = '/Users/munzur/Desktop/FEDERATED_ECG/Data/mitbih_test.csv'
    batch_size = cfg['batch_size']  
    augmentations = Augment([Amplify(), Stretch()])
    train_loader, test_loader, df_train, df_test = create_data_loaders(train_csv, test_csv, batch_size, augmentations=augmentations)
    return train_loader, test_loader, df_train

def data_processing_federated ():

    train_loaders = []
    val_loaders = []
    test_loaders=[]       
    train_loader,test_loader, train_df = data_processing()
    augment = Augment([Amplify(), Stretch()])
    kf = KFold(n_splits=cfg['batch_size'])
    #print(f"Starting K-Fold cross-validation with {cfg['n_splits']} splits")

    for fold_n, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        #print(f"Processing fold {fold_n}")
        train_set = ds(train_df.iloc[train_idx,:-1], train_df.iloc[train_idx,-1], transforms=augment)
        val_set = ds(train_df.iloc[val_idx,:-1], train_df.iloc[val_idx,-1])

        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
        val_loader =   DataLoader(val_set, batch_size=(cfg['batch_size'])*4)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    print ("data loaded success")
    #print (test_loaders)
    return train_loaders,val_loaders, test_loaders

 