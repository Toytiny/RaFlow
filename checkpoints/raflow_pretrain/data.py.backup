#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import ujson
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


class saicDataset(Dataset):
    
    def __init__(self, args, textio, root='/home/toytiny/SAIC_radar/radar_pcs/', partition='train'):
        

        self.npoints = args.num_points
        self.aug = args.aug
        self.partition = partition
        self.root = root + partition+'/'
        self.pc_ls=sorted(os.listdir(self.root),key=lambda x:eval(x.split("/")[-1].split("-")[-1].split(".")[0]))
        self.scene_nbr=int(self.pc_ls[-1].split("-")[1].split("_")[0])
        self.datapath={'sample':[]}
        for idx in range(0,len(self.pc_ls)):
            self.datapath['sample'].append(self.root+self.pc_ls[idx])
            
        textio.cprint(self.partition + ': %d'%len(self.datapath['sample']))
        
    def __getitem__(self, index):
        
    
        sample = self.datapath['sample'][index]
        with open(sample, 'rb') as fp:
            data = ujson.load(fp)
        
        data_1 = data["pc1"]
        data_2 = data["pc2"]
        
        ## obtain groundtruth for multiple tasks during test
        if self.partition =='test':
            trans = np.linalg.inv(np.array(data["trans"]))
            gt = np.array(data["gt"])
            mask = np.array(data["mask"])
        else:
            trans = np.zeros((4,4))
            gt = np.zeros((self.npoints,3))
            mask = np.zeros(self.npoints)

            
        interval = data["interval"]
        pos1=np.vstack((data_1['car_loc_x'],data_1['car_loc_y'],data_1['car_loc_z'])).T.astype('float32')
        pos2=np.vstack((data_2['car_loc_x'],data_2['car_loc_y'],data_2['car_loc_z'])).T.astype('float32')
        vel1=np.array(data_1['car_vel_r']).astype('float32')
        vel2=np.array(data_2['car_vel_r']).astype('float32')
        rcs1=np.array(data_1['rcs']).astype('float32')
        rcs2=np.array(data_2['rcs']).astype('float32')
        power1=np.array(data_1['power']).astype('float32')
        power2=np.array(data_2['power']).astype('float32')
        feature1 = np.vstack((vel1,rcs1,power1)).T
        feature2 = np.vstack((vel2,rcs2,power2)).T
        
        ## downsample to npoints to enable fast batch processing (not in test)
        if self.partition!='test':
            sample_idx1 = np.random.choice(pos1.shape[0], self.npoints, replace=False)
            sample_idx2 = np.random.choice(pos2.shape[0], self.npoints, replace=False)
            
            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            feature1 = feature1[sample_idx1, :]
            feature2 = feature2[sample_idx2, :]
         
        ## data augmentation
        if self.aug and self.partition not in ['test', 'val']  :
            
            T_1 = np.eye(4).astype(np.float32)
            T_2 = np.eye(4).astype(np.float32)
            
            # rotation
            yaw_1,pitch_1,roll_1 = np.random.uniform(-2,2,size=3)
            yaw_2,pitch_2,roll_2 = np.random.uniform(-2,2,size=3)
            angles_1 = [yaw_1, pitch_1,roll_1]
            angles_2 = [yaw_2, pitch_2,roll_2]
            rot1 = R.from_euler('ZYX', angles_1 , degrees=True)
            rot_m1 = rot1.as_matrix()
            rot2 = R.from_euler('ZYX', angles_2 , degrees=True)
            rot_m2 = rot2.as_matrix()
            
            # translation 
            shift_x1, shift_x2 = np.random.uniform(-0.1,0.1,size=2)
            shift_y1, shift_y2 = np.random.uniform(-0.1,0.1,size=2)
            shift_z1, shift_z2 = np.random.uniform(-0.05,0.05,size=2)
            shift_1 = np.array([shift_x1,shift_y1,shift_z1])
            shift_2 = np.array([shift_x2,shift_y2,shift_z2])
            T_1[0:3,0:3] = rot_m1.astype(np.float32)
            T_2[0:3,0:3] = rot_m2.astype(np.float32)
            T_1[0:3,3] = shift_1.astype(np.float32)
            T_2[0:3,3] = shift_2.astype(np.float32)
    
            # apply the random transformation to points
            pos1 = (np.matmul(T_1[0:3, 0:3], pos1.transpose()) + T_1[0:3,3:4]).transpose()
            pos2 = (np.matmul(T_2[0:3, 0:3], pos2.transpose()) + T_2[0:3,3:4]).transpose()
            
            
        return pos1, pos2, feature1, feature2, trans, gt, mask, interval
                
        
    def __len__(self):
        return len(self.datapath['sample'][:100])
    
   
if __name__ == '__main__':
    print('The file can not directly run!!!')
