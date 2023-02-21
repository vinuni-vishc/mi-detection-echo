import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import IPython
from utils.videos import *
data_folders = [
    '/home/vishc2/tuannm/echo/vishc-echo/ext-feat/data_frame',
]

class MIFrameDataset(Dataset):
    def __init__(self, split = 'train', id_fold=0, img_size=(112, 112), clip_length=32, is_crop_lv=False, is_video=True, len_window=5):
        
        self.img_size = img_size
        self.clip_length = clip_length
        self.split = split
        self.is_video = is_video
        self.crop_lv = is_crop_lv
        self.len_window = len_window
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989])
        ])
        
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
    
    def window_items(self, video_path, fr_st=0, len_fr = 5, fr_end=None):
        frames = []
        for id_fr in range(fr_st, fr_st + len_fr):
            file_img = os.path.join(data_folders[0], f'{os.path.basename(video_path[:-4])}__{id_fr:03d}.jpg')
            frame = cv2.imread(file_img)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = crop_and_scale(frame_rgb, self.img_size)
            frame_rgb = frame_rgb / 255.0
            frames.append(frame_rgb)
            # pass
        frames = np.array(frames).transpose((3, 0, 1, 2))
        return frames
        # pass

    
    def get_dataset(self):
        self.data_sets = []
        for idx, ( (fn, nframe), mi) in enumerate(zip(self.video_lists, self.mi_lists)):
            for idy in range(0, int(nframe) - self.len_window):
                self.data_sets.append((fn, idy, mi))
                
    def __getitem__(self, idx):
        
        fn, id_fr, mi = self.data_sets[idx]
        window = self.window_items(fn, id_fr, self.len_window)
        # print(window.shape)
        return window, mi
    
    
 

if __name__ == '__main__':
    types = ['train', 'val', 'test']
    for t in types:
        print(t)
        for w in [5, 7, 9]:
            print(t, w)
            mi = MIFrameDataset(split=t, id_fold=0, len_window=w)
            print(len(mi))    
            cc = mi[0]
            print('**'*30)
        print("--"*30)

    