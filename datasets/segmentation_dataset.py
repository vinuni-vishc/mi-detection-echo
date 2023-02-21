from numpy import mask_indices
from utils.videos import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

seed = 0
random.seed(seed)
np.random.seed(seed)

DIRS_REPLACE = ['/media/tuan/disk4T/VishC/echo/data/HMC-QU-Data/ground-truth',
                '/home/vishc2/tuannm/echo/hmcqu-4c-lv-wall-masks',
                '/home/quanghuy0497/Downloads/tuannm/hmcqu-4c-lv-wall-masks',
                '/media/tuan/disk4T/VishC/echo/data/HMC-QU-Data/ground-truth',]

class LVWallDataset(Dataset):
    def __init__(self,
        data_folder='/home/vishc2/tuannm/echo/hmcqu-4c-lv-wall-masks/masks_jpg',
        split = 'train', img_size=(224, 224),
        id_fold = 0, n_fold=5, transform=None):

        self.img_size = img_size        
        self.data_folds = get_file_segmentation(data_folder)
    
        # self.augmentation = self.__strong_aug(p=0.8)
        self.mi = []
        self.data_fold_imgs_train, self.data_fold_imgs_test = [], []
        
        if split == 'train':
            self.img_list = self.data_folds[id_fold][0]
            self.mi = self.data_folds[id_fold][3]
            self.data_fold_imgs_train = self.data_folds[id_fold][-2]
        else:
            self.img_list = self.data_folds[id_fold][2]
            self.mi = self.data_folds[id_fold][5]
            self.data_fold_imgs_test = self.data_folds[id_fold][-1]
            


    def __len__(self):
        return(len(self.img_list))

    def __getitem__(self, idx):
        file_img, file_mask = self.img_list[idx]
        # print(file_img, file_mask )
        frame = cv2.imread(file_img)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = crop_and_scale(frame_rgb, self.img_size)
        # frame_rgb = frame_rgb / 255.0
        
        
        mask = cv2.imread(file_mask, flags=cv2.IMREAD_GRAYSCALE)    
        mask = crop_and_scale(mask, self.img_size)
        mask = np.array(mask) > 200
        mask = np.expand_dims(mask, axis=0)
        # print(frame_rgb.shape, mask.shape)
        return frame_rgb.transpose((2, 0, 1)), mask


def get_concat_img_video(imgs=[], img_size=(224, 224)):
    img_concat = None
    mask_list = []
    img_msk_list = []
    raw_img_list = []
    
    for i in range(len(imgs)):
        
        
        file_img = imgs[i][0]
        file_mask = imgs[i][1]
        
        # file_img, file_mask = self.img_list[idx]
        # print(file_img, file_mask )
        frame = cv2.imread(file_img)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = crop_and_scale(frame_rgb, img_size)
        # frame_rgb = frame_rgb / 255.0
        
        
        mask = cv2.imread(file_mask, flags=cv2.IMREAD_GRAYSCALE)    
        mask = crop_and_scale(mask, img_size)
        mask = np.array(mask) > 200
        mask = np.expand_dims(mask, axis=0)
        # print(frame_rgb.shape, mask.shape)
        
        # img = frame_rgb.transpose((2, 0, 1))
        
        # img = img.transpose(1, 2, 0).astype(np.uint8)
        
        img = frame_rgb.astype(np.uint8)
        msk = mask.squeeze().astype(np.uint8) * 255
        
        msk_rgb = cv2.cvtColor(msk,cv2.COLOR_GRAY2RGB)
        msk_rgb_0 = msk_rgb.copy()
        
        msk_rgb_02 = msk_rgb.copy()
        
        msk_rgb_02[np.where((msk_rgb_02==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
        
        msk_rgb[np.where((msk_rgb==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
        
        colors_list = np.array([
                [255, 191, 0],
                [16, 161, 157],
                [84, 3, 117],
                [255, 112, 0]
            ])[:,[1, 2, 0]]
        
        
        colors_list = np.array([
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (255, 255, 255),
            (0, 0, 0),
        ])[:,[1, 2, 0]]
        
        
        
        if i == 0:
            msk_rgb_00 = msk_rgb_0.copy()
            msk_rgb_00[np.where((msk_rgb_00==[255, 255, 255]).all(axis=2))] = colors_list[1]
            
            img_msk = cv2.addWeighted(img, 0.5, msk_rgb_00, 0.5, 0.0)
        else:
            msk_rgb_01 = msk_rgb_0.copy()
            msk_rgb_01[np.where((msk_rgb_01==[255, 255, 255]).all(axis=2))] = colors_list[2]
            img_msk = cv2.addWeighted(img, 0.5, msk_rgb_01, 0.5, 0.0)
        
        img_msk_list.append(img_msk)
        raw_img_list.append(img)
        
        if i == 0:
            mask_list.append(msk_rgb)
        else:
            mask_list.append(msk_rgb_02)
            
        if i == 0:
            img_concat = img_msk
        else:
            # new_img = np.array(img_msk.shape)
            # print(img_msk.shape)
            
            white_img = np.ones((img_msk.shape[0], 10, 3), dtype=np.uint8) * 255
            # change black pixel to white
            # mask_list[0] =  np.where((mask_list[0]==[0, 0, 0]).all(axis=2), [255, 255, 255], mask_list[0])
            # mask_list[1] = np.where((mask_list[1]==[0, 0, 0]).all(axis=2), [255, 255, 255], mask_list[1])
            
            
            mask_list[0][np.where((mask_list[0]==[0, 0, 0]).all(axis=2))] = colors_list[1]
            
            # mask_list[1][np.where((mask_list[1]==[0, 0, 0]).all(axis=2))] = colors_list[2]
            
            
            
            # mask_list[0][np.where((mask_list[0]==[0, 0, 255]).all(axis=2))] = colors_list[1]
            
            # mask_list[1][np.where((mask_list[1]==[0, 0, 255]).all(axis=2))] = colors_list[2]
            
            vis_mask = np.ones((img_msk.shape[0], img_msk.shape[1], 3), dtype=np.uint8) * 255

            mask_1_idx = np.where((mask_list[0]==[0, 0, 255]).all(axis=2))
            mask_2_idx = np.where((mask_list[1]==[0, 0, 255]).all(axis=2))
            
            overlap_idx = np.where((mask_list[0]==[0, 0, 255]).all(axis=2) & (mask_list[1]==[0, 0, 255]).all(axis=2))
            only_1_idx = np.where((mask_list[0]==[0, 0, 255]).all(axis=2) & ~(mask_list[1]==[0, 0, 255]).all(axis=2))
            only_2_idx = np.where(~(mask_list[0]==[0, 0, 255]).all(axis=2) & (mask_list[1]==[0, 0, 255]).all(axis=2))
            
            raw_img_list[0][mask_1_idx] = img_msk_list[0][mask_1_idx]
            raw_img_list[1][mask_2_idx] = img_msk_list[1][mask_2_idx]
            
            
            vis_mask[only_1_idx] = colors_list[1]
            vis_mask[only_2_idx] = colors_list[2]
            vis_mask[overlap_idx] = colors_list[3]
            
            # gaussian blur for vis_mask
            # vis_mask = cv2.GaussianBlur(vis_mask, (3, 3), 0)
            # center crop and resize to original image
            # import IPython
            # IPython.embed()
            
            # vis_mask = vis_mask[30:160, 70:180, :]
            # vis_mask = cv2.resize(vis_mask, (img_msk.shape[1], img_msk.shape[0]))
            
            v2 = vis_mask.copy()
            v2 = v2[30:170, 70:180, :]
            v2 = cv2.resize(v2, (img_msk.shape[1], img_msk.shape[0]))
            
            # border_color = [255, 191, 0]
            border_color = [16, 161, 157]
            
            img_concat = cv2.copyMakeBorder(raw_img_list[0], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)
            img_msk = cv2.copyMakeBorder(raw_img_list[1], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)
            img_vis = cv2.copyMakeBorder(v2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)
            
            
            # import IPython
            # IPython.embed()
            
            img_concat = np.concatenate((img_concat , img_msk, img_vis), axis=1)
            img_concat = cv2.cvtColor(img_concat,cv2.COLOR_BGR2RGB)
            # mask_list[1][np.where((mask_list[1]==[0, 0, 255]).all(axis=2))] = colors_list[2]
            
            # msk_rgb[np.where((msk_rgb==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
        
        
            # mask_add_weight = cv2.addWeighted(mask_list[0], 0.5, mask_list[1], 0.5, 0.0)
            
            
            # img_concat = 
            # add border image for img_concat
            # mask_add_weight = cv2.copyMakeBorder(mask_add_weight, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)
            # [255, 191, 0]
            # [16, 161, 157]
            # [84, 3, 117]
            # [255, 112, 0]
            
            # concatentate the images with vertical space
            # img_concat = cv2.vconcat([img_concat, img_msk, mask_add_weight])
            
    return img_concat

if __name__ == '__main__':
    
    ds = LVWallDataset(split='train', id_fold=0)
    
    print(len(ds))
    print(ds.mi, len(ds.data_fold_imgs_train), len(ds.data_fold_imgs_test))
    
    # show mi
    cnt_mi, cnt_non_mi = 0, 0
    limit_cnt = 4
    for vid_id in range(len(ds.mi)):
        # print(vid_id)
        mi = ds.mi[vid_id]
        # if mi == 1:
        #     cnt_mi += 1
        #     if cnt_mi > limit_cnt:
        #         continue
        # else:
        #     cnt_non_mi += 1
        #     if cnt_non_mi > limit_cnt:
        #         continue
        
        print(vid_id, mi)
        text_mi = 'MI' if mi == 1 else 'NON-MI'
        print("Select file: ", ds.data_fold_imgs_train[vid_id][0], ds.data_fold_imgs_train[vid_id][-1])
        
        img_concat = get_concat_img_video([ ds.data_fold_imgs_train[vid_id][0], ds.data_fold_imgs_train[vid_id][-1] ]  )
        cv2.imwrite(f'/home/vishc2/tuannm/echo/vishc-echo/datasets/tmp/dataset_isbi_1212_{vid_id:02d}_{text_mi}.png', img_concat)
    # import IPython
    # IPython.embed()
    
    # exit(0)
    
    
    
    img_col = []
    img_row = []
    cnt = 0
    for idm in range(215, 300, 3):
        print(idm, cnt)
        img = ds[idm][0].transpose(1, 2, 0).astype(np.uint8)
        msk = ds[idm][1].squeeze().astype(np.uint8) * 255
        # print(msk.shape)
        msk_rgb = cv2.cvtColor(msk,cv2.COLOR_GRAY2RGB)
        msk_rgb[np.where((msk_rgb==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
        
        # print(idm, img.shape, msk_rgb.shape)
        img_msk = cv2.addWeighted(img, 0.7, msk_rgb, 0.3, 0.0)
        # print(img_msk.shape, np.max(img), np.min(img))
        
        
        if cnt % 5 == 0:    
            img_row = img_msk
            
        else:
            img_row = np.concatenate((img_row, img_msk), axis=1)
        
        if cnt % 5 == 4:
            if len(img_col) == 0:
                img_col = img_row
            else:
                img_col = np.concatenate((img_col, img_row), axis=0)
      
        cnt += 1

        if cnt == 15:
            break
        
        # if cnt == 5:
        #     plt.imshow(img_row)
        #     plt.show()
        #     exit(0)
        # # break
    
    # plt.imshow(img_col)
    # plt.show()
    cv2.imwrite('/home/vishc2/tuannm/echo/vishc-echo/datasets/tmp/dataset_isbi_3v.png', img_col)
   
    # d0 = ds[0]
    # print(d0[1].shape)
    # plt.imshow(d0[0].transpose(1, 2, 0))
    # plt.show()
    
    # plt.imshow(d0[1][0], cmap='gray')
    # plt.show()
