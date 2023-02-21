import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import IPython
from utils.videos import *
data_folders = [
    '/home/vishc2/tuannm/echo/vishc-echo/ext-feat/data_frame',
]

class MIVideoDataset(Dataset):
    def __init__(self, split = 'train', id_fold=0, img_size=(112, 112), is_crop_lv=False, is_video=True, len_window=5):
        
        self.img_size = img_size
        self.split = split
        self.is_video = is_video
        self.crop_lv = is_crop_lv
        self.len_window = len_window
        
        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989])
        # ])
        
        self.dataX, self.dataY, self.data_folds = get_files_avi()
        self.data_sets = []
        
        if split == 'train':
            self.video_lists, self.mi_lists = self.data_folds[id_fold][0], self.data_folds[id_fold][3]
        elif split == 'val':
            self.video_lists, self.mi_lists = self.data_folds[id_fold][1], self.data_folds[id_fold][4]
        elif split == 'test':
            self.video_lists, self.mi_lists = self.data_folds[id_fold][2], self.data_folds[id_fold][5]
            
        self.get_dataset()
        print("Number of instance in dataset: {}".format(len(self.data_sets)))
        print(f"Number of video in fold: {id_fold} data split {split} len: {len(self.video_lists)}")
        
    def __len__(self):
        return len(self.data_sets)
    
    def window_items(self, video_path, n_fr=32, len_window = 5, fr_end=None):
        clip = read_file_video(video_path, res=self.img_size, max_len=self.len_window, bbox=None)
        clip = clip / 255.0
        while len(clip) < self.len_window:
            clip = np.concatenate((clip, np.expand_dims(clip[-1], axis=0) ), axis=0)
        return np.transpose(clip, (3, 0, 1, 2))    

    
    def get_dataset(self):
        self.data_sets = []
        for idx, ( (fn, nframe), mi) in enumerate(zip(self.video_lists, self.mi_lists)):
            # for idy in range(0, int(nframe) - self.len_window):
            self.data_sets.append((fn, nframe, mi))
                
    def __getitem__(self, idx):
        fn, n_fr, mi = self.data_sets[idx]
        # return fn, mi
        window = self.window_items(fn, n_fr, self.len_window)
        return window, mi
    
    
 

if __name__ == '__main__':
    types = ['train', 'val', 'test']
    for f in range(1):
        for t in types:
            print(f, t)
            mi = MIVideoDataset(split=t, id_fold=f, len_window=32)
            for i, m in enumerate(mi):
                print(i, m)
            print("--"*30)

    