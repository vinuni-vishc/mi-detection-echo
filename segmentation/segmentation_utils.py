import torch
from easydict import EasyDict

import segmentation_models_pytorch as smp
import numpy as np
import cv2


def get_loss(ypred, ytrue, batch):
    '''compute loss tensor from prediction and target and additional batch information, to be override by other Trainer'''
    loss = torch.nn.BCEWithLogitsLoss()
    loss = loss(ypred, ytrue) # - torch.log( mIoU(torch.sigmoid(ypred), ytrue) )
    return loss

# def get_metric(loss, iou, bsize):
#     return EasyDict(dict(loss=loss.item(), iou=iou.item(), bsize=bsize))

# def get_mean(metrics):
#     '''compute the average of metrics in a list of EasyDict(...) '''
#     return EasyDict({key : np.sum([x[key] * x['bsize'] for x in metrics] / np.sum([x['bsize'] for x in metrics]) ) for key in ['loss', 'iou']})

def mIoU(y_pred, masks):
    '''compute mean IoU loss approximately using prediction and target'''
    y_pred, y_true = y_pred.float(), torch.round(masks)
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def get_iou(ypred, ytrue):
    '''compute iou metric from prediction and target, to be override by other Trainer'''
    # return CriticIoU(torch.sigmoid(ypred), ytrue)
    return mIoU(ypred,ytrue)  


def get_model_and_optim(config, device=None):
    # print(get_model_and_optim, config)
    architecture = config.architecture
    if architecture == 'Unet':
        model = smp.Unet(config.encoder, encoder_weights='imagenet',
                    in_channels=3, classes=1).to(device)
    elif architecture == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(config.encoder, encoder_weights='imagenet',
                    in_channels=3, classes=1).to(device)
    elif architecture == 'Linknet':
        model = smp.Linknet(config.encoder, encoder_weights='imagenet',
                    in_channels=3, classes=1).to(device)
    elif architecture == 'FPN':
        model = smp.FPN(config.encoder, encoder_weights='imagenet',
                    in_channels=3, classes=1).to(device)
    elif architecture == 'PAN':
        model = smp.PAN(config.encoder, encoder_weights='imagenet',
                    in_channels=3, classes=1).to(device)
    elif architecture == 'DeepLabV3':
        model = smp.DeepLabV3(config.encoder, encoder_weights='imagenet',
                    in_channels=3, classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def segment_coordinates(point1_y = None,point2_y = None,point3_y = None): 
    # Divide the LV wall into 7 segments::
    
    # SEGMENT DESCRIPTIONS:
    # --------------------
    # SEG1: 2L/7
    # SEG2: 2L/7
    # SEG3: L/7
    # SEG4: 2L/7 & 2R/7
    # SEG5: R/7
    # SEG6: 2R/7
    # SEG7: 2R/7
    ##
    L = point1_y - point3_y
    
    R = point2_y - point3_y
    
    #Find the each segments lower and upper point respectively::
    Lseg1 = np.array([point1_y, (point1_y - (2 * L / 7))])
    Lseg2 = np.array([(point1_y - (2 * L / 7)),(point1_y - (4 * L / 7))])
    Lseg3 = np.array([(point1_y - (4 * L / 7)),(point1_y - (5 * L / 7))])
    Lseg4 = np.array([(point1_y - (5 * L / 7)),point3_y])
    Rseg1 = np.array([point2_y, (point2_y - (2 * R / 7))])
    Rseg2 = np.array([(point2_y - (2 * R / 7)),(point2_y - (4 * R / 7))])
    Rseg3 = np.array([(point2_y - (4 * R / 7)),(point2_y - (5 * R / 7))])
    Rseg4 = np.array([(point2_y - (5 * R / 7)),point3_y])
    #Left and Right coordinates are ready::
    # segments = 
    s0 = np.array([[Lseg1],[Lseg2],[Lseg3],[Lseg4]])
    s1 = np.array([[Rseg1],[Rseg2],[Rseg3],[Rseg4]])
    return np.array([s0, s1])


def get_tracked_points(curves):
    points = []
    for idc, curve in enumerate(curves):
        y1, y2, y3, y4, out_left_poly, in_left_poly, in_right_poly, out_right_poly, x1, x2, x3, x4 = curve
      
        los = segment_coordinates(y1, y3, y4)
        
        ylin = [np.linspace(los[0][id][0][0], los[0][id][0][1], 5) for id in range(4)]
        bl1 = [(y, out_left_poly(y), in_left_poly(y), int(y), out_left_poly(int(y)), in_left_poly(int(y)) ) for y in ylin[0]]
        bl2 = [(y, out_left_poly(y), in_left_poly(y), int(y), out_left_poly(int(y)), in_left_poly(int(y)) ) for y in ylin[1]]
        bl3 = [(y, out_left_poly(y), in_left_poly(y), int(y), out_left_poly(int(y)), in_left_poly(int(y)) ) for y in ylin[2]]
        bl41 = [(y, out_left_poly(y), in_left_poly(y), int(y), out_left_poly(int(y)), in_left_poly(int(y)) ) for y in ylin[3]]
        
        
        yrin = [np.linspace(los[1][id][0][0], los[1][id][0][1], 5) for id in range(4)]
        br42 = [(y, out_right_poly(y), in_right_poly(y), int(y), out_right_poly(int(y)), in_right_poly(int(y))) for y in yrin[3]]
        br5 = [(y, out_right_poly(y), in_right_poly(y), int(y), out_right_poly(int(y)), in_right_poly(int(y)) ) for y in yrin[2]]
        br6 = [(y, out_right_poly(y), in_right_poly(y), int(y), out_right_poly(int(y)), in_right_poly(int(y)) ) for y in yrin[1]]
        br7 = [(y, out_right_poly(y), in_right_poly(y), int(y), out_right_poly(int(y)), in_right_poly(int(y)) ) for y in yrin[0]]
        
        # br42 = [(y, in_right_poly(y), out_right_poly(y), int(y), in_right_poly(int(y)), in_right_poly(int(y))) for y in yrin[3]]
        # br5 = [(y, out_right_poly(y), in_right_poly(y), int(y), out_right_poly(int(y)), in_right_poly(int(y)) ) for y in yrin[2]]
        # br6 = [(y, out_right_poly(y), in_right_poly(y), int(y), out_right_poly(int(y)), in_right_poly(int(y)) ) for y in yrin[1]]
        # br7 = [(y, out_right_poly(y), in_right_poly(y), int(y), out_right_poly(int(y)), in_right_poly(int(y)) ) for y in yrin[0]]
        
        
        bllr = [bl1, bl2, bl3, bl41, br42, br5, br6, br7]
        points.append(bllr)
    return np.array(points)

def get_motion_L2(points, id_st = 0, end_frame=-1):
    if end_frame == -1:
        end_frame = len(points)
    
    points = points[id_st:end_frame]
    
    tracked_points_D_L2 = np.take(points, [3, 5], axis=3)
    D_L2 = []
    tracked_points_D_L2 = np.take(points, [3, 5], axis=3)
    for tracked_point in tracked_points_D_L2:
        seg1, seg2, seg3, seg41, seg42, seg5, seg6, seg7 = tracked_point
        # d17 = np.sum( np.sqrt(np.take(seg1 - seg7, (0, 1), axis=1)**2), axis=-1 )[2]
        # d26 = np.sum( np.sqrt(np.take(seg2 - seg6, (0, 1), axis=1)**2), axis=-1 )[2]
        # d35 = np.sum( np.sqrt(np.take(seg3 - seg5, (0, 1), axis=1)**2), axis=-1 )[2]
        d17 = np.sum( np.sqrt((seg1 - seg7)**2), axis=-1 )[2]
        d26 = np.sum( np.sqrt((seg2 - seg6)**2), axis=-1 )[2]
        d35 = np.sum( np.sqrt((seg3 - seg5)**2), axis=-1 )[2]
        D_L2.append([d17, d26, d35])
        
    D_L2 = np.array(D_L2)


    id_st = 0
    tracked_points = np.take(points, [0, 2], axis=3)
    st_seg1, st_seg2, st_seg3, st_seg41, st_seg42, st_seg5, st_seg6, st_seg7 = tracked_points[id_st]
    disp_frames = [ [0, 0, 0, 0, 0, 0] ]

    for cur_frame in tracked_points[id_st + 1: len(tracked_points) ]:
        # st_seg1, st_seg2, st_seg3, st_seg41, st_seg42, st_seg5, st_seg6, st_seg7 = cur_frame
        cur_seg1, cur_seg2, cur_seg3, cur_seg41, cur_seg42, cur_seg5, cur_seg6, cur_seg7 = cur_frame
        
        disp_1 = np.mean(np.sqrt( np.sum((cur_seg1 - st_seg1)[:, :2]**2, axis = 1)))
        disp_2 = np.mean(np.sqrt( np.sum((cur_seg2 - st_seg2)[:, :2]**2, axis = 1)))
        disp_3 = np.mean(np.sqrt( np.sum((cur_seg3 - st_seg3)[:, :2]**2, axis = 1)))
        
        disp_5 = np.mean(np.sqrt( np.sum((cur_seg5 - st_seg5)[:, :2]**2, axis = 1)))
        disp_6 = np.mean(np.sqrt( np.sum((cur_seg6 - st_seg6)[:, :2]**2, axis = 1)))
        disp_7 = np.mean(np.sqrt( np.sum((cur_seg7 - st_seg7)[:, :2]**2, axis = 1)))
        
        disp_frames.append([disp_1, disp_2, disp_3, disp_7, disp_6, disp_5])
    disp_frames = np.array(disp_frames)

    minD_L2 = np.min(D_L2, axis=0)
    minD_L2 = np.array([minD_L2[0], minD_L2[1], minD_L2[2], minD_L2[2], minD_L2[1], minD_L2[0] ])
    motion_L2 = np.max(disp_frames, axis=0)
    motion_L2 = motion_L2 / minD_L2
    return motion_L2, disp_frames, D_L2


def get_curve_from_contours(w, h, motions):
    curves = []
    contours_frames = []
    
    for idm, motion in enumerate(motions):
        # motion = cv2.re
        motion = cv2.resize(motion, (w, h))
        
        contours, hierarchy = cv2.findContours(motion.astype(np.uint8).copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        n_max_point = 0
        max_countour = []
        for cc in contours:
            if len(cc) > n_max_point:
                n_max_point = len(cc)
                max_countour = cc
        # print("Shape max_coutour: {}".format(len(max_countour)))
        # contours = [max_countour]
        bp = max_countour.reshape(-1, 2)
        # print(bp.shape)
        
        # return
        py_min = np.argmin(np.abs(bp[:, 1]))
        bp_roll = np.roll(bp, -py_min, axis=0)
        
        contours_frames.append(bp_roll)
        
        data_y = bp_roll[:, 1]
        
        # data_y = bp_roll[:, 1]
        n_dy = len(data_y)
        p1 = 0
        
        # np.argwhere(bp_roll[-10:10, 1]==319)
        # p2 = np.where(data_y == np.amax(data_y))
        d2 = data_y[:n_dy//2]
        idx2 = np.where(d2 == np.amax(d2))
        p2 = idx2[0][len(idx2) // 2]

        d4 = data_y[n_dy//2 : n_dy]
        idx4 = np.where(d4 == np.amax(d4))
        p4 = n_dy//2 + idx4[0][len(idx4) // 2]
        # p4 = n_dy//2 + np.argmax()
        
        p3 = p2 + np.argmin(np.sum((bp_roll[p2:p4] - bp_roll[p1] ) ** 2, axis=1))

        # print("frame id: {} peaks_max: {} peaks_min: {}  shape_y: {} p: {}".format(idm, peaks_max, peaks_min, len(data_y), [p1, p2, p3, p4]))
        # print(p1, p2, p3, p4)
        
        bp_out_lef = bp_roll[p1:p2]
        out_left_poly = np.polyfit(bp_out_lef[:, 1], bp_out_lef[:, 0], 4)
        out_left_poly = np.poly1d(out_left_poly)
        
        # out_left_y = get_range_incrementals(bp_roll[p1][1], bp_roll[p2][1])
        # out_left_x = out_left_poly(out_left_y).astype(int)
        # print("Go after p1 p2")
        # if len(max_countour) <650:
        #     import IPython
        #     IPython.embed()
            
        bp_in_lef = bp_roll[p2:p3]
        in_left_poly = np.polyfit(bp_in_lef[:, 1], bp_in_lef[:, 0], 4)
        in_left_poly = np.poly1d(in_left_poly)
        # in_left_y = get_range_incrementals(bp_roll[p3][1], bp_roll[p2][1])
        # in_left_x = in_left_poly(in_left_y).astype(int)
        # print("Go after p2 p3")
        
        bp_in_right = bp_roll[p3:p4]
        in_right_poly = np.polyfit(bp_in_right[:, 1], bp_in_right[:, 0], 4)
        in_right_poly = np.poly1d(in_right_poly)
        # in_right_y = get_range_incrementals(bp_roll[p3][1], bp_roll[p4][1])
        # in_right_x = in_right_poly(in_right_y).astype(int)

        bp_out_right = bp_roll[p4:]
        out_right_poly = np.polyfit(bp_out_right[:, 1], bp_out_right[:, 0], 4)
        out_right_poly = np.poly1d(out_right_poly)
        # out_right_y = get_range_incrementals(bp_roll[-1][1], bp_roll[p4][1])
        # out_right_x = out_right_poly(out_right_y).astype(int)
        curves.append(
            [   bp_roll[p2][1], bp_roll[p3][1], bp_roll[p4][1], bp_roll[p1][1], 
                out_left_poly, in_left_poly, in_right_poly, out_right_poly,
                bp_roll[p2][0], bp_roll[p3][0], bp_roll[p4][0], bp_roll[p1][0], 
            ]
        )
    return curves, contours_frames