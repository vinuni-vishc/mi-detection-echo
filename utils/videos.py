import os
import glob
import cv2
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

seed = 0
random.seed(seed)
np.random.seed(seed)


# from collections import Counter
# from torchvision import transforms
# from albumentations import (
#     HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
#     Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
#     GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, GaussNoise,
#     Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose
# )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def read_all_img_in_folder(folder_path='/home/vishc2/tuannm/echo/vishc-echo/ext-feat/data_frame'):
    img_list = {}
    for file_name in glob.glob(os.path.join(folder_path, '**', '*.jpg'), recursive=True):
        img = cv2.imread(file_name)
        img_list[file_name] = img_list
    print(f"Number of images in folder: {len(img_list)}")
    return img_list


def crop_and_scale(img: np.ndarray, res=(640, 480), bbox=None):
    
    if bbox is not None:
        x_min, x_max, y_min, y_max = bbox['x_min'], bbox['x_max'], bbox['y_min'], bbox['y_max']
        img = img[y_min: y_max, x_min: x_max]
            
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        # giam do rong (h)
        padding = int( ((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
        
    if r_in < r_out:
        # giam do cao (w)
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    
    img = cv2.resize(img, res)
    return img


def read_file_video(path, res=None, max_len=None, bbox=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        if max_len is not None and i >= max_len:
            break
        i += 1
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if res is not None:
            frame = crop_and_scale(frame, res, bbox)
        
        frames.append(frame)
        
    cap.release()

    return np.array(frames[:max_len])

def get_data_mi(fn_csv='/home/vishc2/tuannm/echo/vishc-echo/src/lv-motion/data_cycle_292_0307.csv'):

    pass

    

def get_files_avi(fn_csv='/home/vishc2/tuannm/echo/vishc-echo/src/lv-motion/data_cycle_292_0307.csv',
                  displacements_file='/home/vishc2/tuannm/echo/vishc-echo/src/pipeline/matlab/ap-mi-python/saved_2c_4c_tracking_20200326.json',
                  n_fold=5):
    
    data_mi = pd.read_csv(fn_csv)
    data_videos = {}
    for idx, (_, dm) in enumerate(data_mi.iterrows()):
        data_videos[dm['fn'].split('/')[-1][:-4]] =  dm # {'ref': dm['ref'], 'end': dm['end'], 'nframe': dm['nframe']}
        
    dataX, dataY, dataYs = [], [], []
    
    for idf, (fnk, vn) in enumerate(data_videos.items()):
        # if 'A4C' not in vn['fn']:
        #     continue
        mi = 1 if (vn['s1'] + vn['s2'] + vn['s3'] + vn['s4'] + vn['s5'] + vn['s6']) > 0 else 0
        fn = vn['fn'].replace('/Volumes/tuan/vinuni/echo/vishc-echo/data/HMC-QU-Data/videos/', '/home/vishc2/tuannm/echo/vishc-echo/data/archive/HMC-QU/')
        if os.path.isfile(fn): # and vn['nframe'] >= 32:
            dataX.append([fn, int(vn['nframe']) ])
            dataYs.append(mi)
            dataY.append([vn['s1'], vn['s2'], vn['s3'], vn['s4'], vn['s5'], vn['s6']  ]) 
                
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataYs = np.array(dataYs)

    with open(displacements_file, 'r') as fr:
        
        motion_L2 = json.load(fr)
    
    data_motion = {}
    for k, v in motion_L2.items():
        data_motion[k] = np.array(v['motion_L2'][0])
    
    
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    
    data_folds = []
    
    for id_fold, (train_index, test_index) in enumerate(skf.split(dataX, dataYs)):
        
        x_train_fold = dataX[train_index]
        x_test_fold = dataX[test_index]
        y_train_fold = dataYs[train_index]
        y_test_fold = dataYs[test_index]
        
        X_train, X_test, y_train, y_test = train_test_split(x_train_fold, y_train_fold, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(x_train_fold, y_train_fold, test_size=0.2, random_state=42)
        # X_train, y_train = x_train_fold, y_train_fold
        X_test, y_test = x_test_fold, y_test_fold
        # X_val, y_val = x_test_fold, y_test_fold
                    
        # data_folds.append([X_train, X_val, X_test, y_train, y_val, y_test])
        data_folds.append([x_train_fold, X_val, X_test, y_train_fold, y_val, y_test])
        
        # print("fold number: {} len training: {} val: {} test: {}".format(id_fold, len(X_train), len(X_val), len(X_test)))    
            
    return dataX, dataY, data_folds    



def extend_frame(clip, clip_length=32):
    while len(clip) < clip_length:
        clip = np.concatenate((clip, np.expand_dims(clip[-1], axis=0) ), axis=0)
    return np.transpose(clip, (3, 0, 1, 2))
        
    # if self.is_video:
    
    # clip = np.mean(clip, axis=0)
    # return np.transpose(clip, (2, 0, 1))
    
def get_file_segmentation(mask_folder='/home/vishc2/tuannm/echo/hmcqu-4c-lv-wall-masks/masks_jpg', 
                          fn_csv='/home/vishc2/tuannm/echo/vishc-echo/src/lv-motion/data_cycle_292_0307.csv',
                          n_fold=5):
    # print("start fun get_file_segmentation")
    files_img, files_mask = [], []
    # print(glob.glob(os.path.join(mask_folder, '**', '*.jpg'), recursive=True))
    for idf, fm in enumerate(glob.glob(os.path.join(mask_folder, '**', '*.jpg'), recursive=True)):
        fim = fm.replace('masks_jpg', 'images_jpg')
        # print(idf, fm, fim)
        # break
        if os.path.isfile(fim):
            files_img.append(fim)
            files_mask.append(fm)
    # print("Number of images: {}".format(len(files_img)))
    data_masks = {}
    for idx, (f1, f2) in enumerate(zip(files_img, files_mask)):
        # id_patient = f1.split('/')[-1].split('__')[0].split(' ')[0].split('_')[0]
        id_patient = f1.split('/')[-1].split('__')[0]
        # print(idx, f1, f2, id_patient)
        # break
        # break
        if id_patient not in data_masks:
            data_masks[id_patient] = []
        data_masks[id_patient].append([f1, f2])

    
    # print(data_masks.keys())
    
    folder_videos = '/home/vishc2/tuannm/echo/vishc-echo/data/archive/HMC-QU'    
    files_avi = glob.glob(os.path.join(folder_videos, '*A4C*', '*.avi'), recursive=True)
    
    video_map = {}
    cnt_found = 0
    for fp in files_avi:
        id_patient = fp.split('/')[-1].split('__')[0][:-4]
        # print(id_patient)
        if id_patient not in video_map:
            video_map[id_patient] = fp
    
    # for k in data_masks.keys():
    #     if k not in video_map:
    #         print("k: {} not in video_map".format(k))
    #     else:
    #         print("Found k: {}".format(k))
    #         cnt_found += 1
            
    # print(cnt_found)
    
    # exit(0)
    data_mi = pd.read_csv(fn_csv)
    data_videos = {}
    # data_mi = {}
    dataX, dataY = [], []
    for idx, (_, dm) in enumerate(data_mi.iterrows()):
        id_patient = dm['fn'].split('/')[-1][:-4]
        data_videos[id_patient] =  dm # {'ref': dm['ref'], 'end': dm['end'], 'nframe': dm['nframe']}
        mi = 1 if (dm['s1'] + dm['s2'] + dm['s3'] + dm['s4'] + dm['s5'] + dm['s6']) > 0 else 0
        # print(idx, id_patient, mi)
        # break
        if id_patient in data_masks:
        # print(idx, id_patient, mi)
            # data_mi[id_patient] = mi
            dataX.append(id_patient)
            dataY.append(mi)
    
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    # print(len(dataX), len(dataY))
        
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    
    data_folds = []
    video_folds = []
    
    for id_fold, (train_index, test_index) in enumerate(skf.split(dataX, dataY)):
        
        x_train_fold = dataX[train_index]
        x_test_fold = dataX[test_index]
        y_train_fold = dataY[train_index]
        y_test_fold = dataY[test_index]
        # print("process to id_fold: {}".format(id_fold))
        # data_fold = [data_masks[p] for p in x_train_fold]
        # import IPython
        # IPython.embed()
        data_fold_train, data_fold_test = [], []
        video_fold_train, video_fold_test = [], []
        data_fold_imgs_train, data_fold_imgs_test = [], []
        
        for p in x_train_fold:
            video_fold_train.append(video_map[p])
            dp = sorted(data_masks[p], key=lambda x: int(x[0].split('/')[-1][:-4].split('__')[-1]))
            data_fold_imgs_train.append(dp)
            for p2 in dp:
                data_fold_train.append(p2)
                
        for p in x_test_fold:
            video_fold_test.append(video_map[p])
            dp = sorted(data_masks[p], key=lambda x: int(x[0].split('/')[-1][:-4].split('__')[-1]))
            data_fold_imgs_test.append(dp)
            for p2 in dp:
                data_fold_test.append(p2)
        
        # data_folds.append([data_fold_train, [], data_fold_test, y_train_fold, [], y_test_fold])
        data_folds.append([data_fold_train, video_fold_train, data_fold_test, y_train_fold, video_fold_test, y_test_fold, data_fold_imgs_train, data_fold_imgs_test])
        # video_folds.append([video_fold_train, video_fold_test])
        from collections import Counter
        
        print("fold number: {} len training: {} val: {} test: {}".format(id_fold, len(data_fold_train), len([]), len(data_fold_test)))
        print("train:", Counter(y_train_fold), "test:", Counter(y_test_fold))
        print("train:", len(video_fold_train), "test:", len(video_fold_test))
        
        
    return data_folds


# def __do_augmentation(self, image, mask):
#     '''use albumentations for data augmentation'''
#     data = {"image": image, "mask": mask}
#     augmented = self.augmentation(**data)
#     return augmented['image'], augmented['mask']

# def __strong_aug(self, p=0.5):
#     '''preset augmentation'''
#     return Compose([
#         # RandomRotate90(),
#         # Flip(),
#         # Transpose(),
#         OneOf([
#             GaussNoise(),
#             GaussNoise(),
#         ], p=0.2),
#         OneOf([
#             MedianBlur(blur_limit=3, p=0.1),
#             Blur(blur_limit=3, p=0.1),
#         ], p=0.2),
#         # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.2),
#         # OneOf([
#         #     OpticalDistortion(p=0.3),
#         #     GridDistortion(p=0.1),
#         #     PiecewiseAffine(p=0.3),
#         # ], p=0.2),
#         # OneOf([
#         #     CLAHE(clip_limit=2),
#         #     Sharpen(),
#         #     Emboss(),
#         #     RandomBrightnessContrast(),
#         # ], p=0.3),
#         # HueSaturationValue(p=0.3),
#     ], p=p, additional_targets=dict(
#         image1="image", image2="image", image3="image", image4="image", image5="image",
#         mask1="mask", mask2="mask", mask3="mask", mask4="mask", mask5="mask"
#     ))
    
if __name__ == '__main__':
    get_file_segmentation()