#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from losses import *


def train_one_epoch(args, net, train_loader, opt, mode):
    
    if mode=='train':
        net.train()
    elif mode=='val':
        net.eval()
        
    num_examples = 0
    total_loss = 0
  

    if args.model=='raflow':
        loss_items={
            'Loss': [],
            'chamferLoss': [],
            'veloLoss':[],
            'smoothnessLoss':[],
            }
    
        
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        
        pc1, pc2, ft1, ft2, _, gt , mask, interval= data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        ft1 = ft1.cuda().transpose(2,1).contiguous()
        ft2 = ft2.cuda().transpose(2,1).contiguous()
        mask = mask.cuda()
        interval = interval.cuda().float()
        gt = gt.cuda().float()
        
        batch_size = pc1.size(0)
        num_examples += batch_size
        vel1 = ft1[:,0]

        
        if args.model=='raflow':
            _, agg_f, _,_ = net(pc1, pc2, ft1, ft2, interval)
            loss, items = computeloss(pc1,pc2, agg_f, vel1,interval, args) 
        
        if mode=='train':    
            opt.zero_grad() 
            loss.backward()
            opt.step()
        
        total_loss += loss.item() * batch_size
        
        
        for l in loss_items:
            loss_items[l].append(items[l]) 

        
    total_loss=total_loss*1.0/num_examples
    
    for l in loss_items:
        loss_items[l]=np.mean(np.array(loss_items[l]))
    
    return total_loss, loss_items
    

def plot_loss_epoch(train_items_iter, args, epoch):
    
    plt.clf()
    plt.plot(np.array(train_items_iter['Loss']).T, 'b')
    plt.plot(np.array(train_items_iter['chamferLoss']).T, 'r')
    plt.plot(np.array(train_items_iter['veloLoss']).T, 'g')
    plt.plot(np.array(train_items_iter['smoothnessLoss']).T, 'c')
    plt.legend(['Total','chamferLoss','veloLoss','smoothness'], loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('checkpoints/%s/loss_train_%s.png' %(args.exp_name,epoch),dpi=500)
 
    

def get_carterian_res(pc, sensor, args):

    ## measure resolution for r/theta/phi
    if sensor == 'radar': 
        if args.dataset == 'saicDataset': # LRR30
            r_res = 0.2 # m
            theta_res = 1 * np.pi/180 # radian
            phi_res = 1.6 *np.pi/180  # radian
        if args.dataset == 'vodDataset': # ZF FRGen21
            r_res: 0.2 # m
            theta_res: 1.5 * np.pi/180, # radian
            phi_res: 1.5 *np.pi/180  # radian
                
        
    if sensor == 'lidar': # HDL-64E
        r_res = 0.04 # m
        theta_res = 0.4 * np.pi/180 # radian
        phi_res = 0.08 *np.pi/180  # radian
         
    res = np.array([r_res, theta_res, phi_res])
    ## x y z
    x = pc[:,0]
    y = pc[:,1]
    z = pc[:,2]
    
    ## from xyz to r/theta/phi (range/elevation/azimuth)
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arcsin(z/r)
    phi = np.arctan2(y,x)
    
    ## compute xyz's gradient about r/theta/phi 
    grad_x = np.stack((np.cos(phi)*np.cos(theta), -r*np.sin(theta)*np.cos(phi), -r*np.cos(theta)*np.sin(phi)),axis=2)
    grad_y = np.stack((np.sin(phi)*np.cos(theta), -r*np.sin(phi)*np.sin(theta), r*np.cos(theta)*np.cos(phi)),axis=2)
    grad_z = np.stack((np.sin(theta), r*np.cos(theta), np.zeros((np.size(x,0),np.size(x,1)))),axis=2)
    
    ## measure resolution for xyz (different positions have different resolution)
    x_res = np.sum(abs(grad_x) * res,axis=2)
    y_res = np.sum(abs(grad_y) * res,axis=2)
    z_res = np.sum(abs(grad_z) * res,axis=2)
    
    xyz_res = np.stack((x_res,y_res,z_res),axis=2)
    
    return xyz_res
        
def eval_scene_flow(pc, pred, labels, mask, args):
    
    pc = pc.cpu().numpy()
    pred = pred.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    mask = mask.cpu().numpy()
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)
    epe = np.mean(error)
    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) 

    ## obtain x y z measure resolution for each point (radar lidar)
    xyz_res_r = get_carterian_res(pc, 'radar', args) 
    res_r = np.sqrt(np.sum(xyz_res_r,2)+1e-20)
    xyz_res_l = get_carterian_res(pc, 'lidar', args) 
    res_l = np.sqrt(np.sum(xyz_res_l,2)+1e-20)
    
    ## calcualte Resolution Normalized Error
    re_error = error/(res_r/res_l)
    rne = np.mean(re_error)
    mov_rne = np.sum(re_error[mask==0])/(np.sum(mask==0)+1e-6)
    stat_rne = np.mean(re_error[mask==1])
    avg_rne = (mov_rne+stat_rne)/2
    
    ## calculate Strict/Relaxed Accuracy Score
    sas = np.sum(np.logical_or((re_error <= 0.10), (re_error/gtflow_len <= 0.10)))/(np.size(pred,0)*np.size(pred,1))
    ras = np.sum(np.logical_or((re_error <= 0.20), (re_error/gtflow_len <= 0.20)))/(np.size(pred,0)*np.size(pred,1))
    
    sf_metric = {'rne':rne, '50-50 rne': avg_rne, 'mov_rne': mov_rne, 'stat_rne': stat_rne,\
                 'sas': sas, 'ras': ras, 'epe':epe}
    
    return sf_metric

    
def eval_motion_seg(pre, gt):
    
    pre = pre.cpu().detach().numpy()
    gt = gt.cpu().numpy()
    tp = np.logical_and((pre==1),(gt==1)).sum()
    tn = np.logical_and((pre==0),(gt==0)).sum()
    fp = np.logical_and((pre==1),(gt==0)).sum()
    fn = np.logical_and((pre==0),(gt==1)).sum()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = tp/(tp+fn)
    miou = 0.5*(tp/(tp+fp+fn+1e-4)+tn/(tn+fp+fn+1e-4))
    seg_metric = {'acc': acc, 'miou': miou, 'sen': sen}
    
    return seg_metric
