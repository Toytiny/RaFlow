import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

    
def transform_to_ego(pc,T):
    
    pos = (np.matmul(T[0:3, 0:3], pc) + T[0:3,3:4])
    
    return pos


def get_matrix_from_ext(ext):
    
    N = np.size(ext,0)
    if ext.ndim==2:
        rot = R.from_euler('ZYX', ext[:,3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((N,4,4))
        tr[:,:3,:3] = rot_m
        tr[:,:3, 3] = ext[:,:3]
        tr[:, 3, 3] = 1
    if ext.ndim==1:
        rot = R.from_euler('ZYX', ext[3:], degrees=True)
        rot_m = rot.as_matrix()
        tr = np.zeros((4,4))
        tr[:3,:3] = rot_m
        tr[:3, 3] = ext[:3]
        tr[ 3, 3] = 1
    return tr

    
def visulize_result_2D(pc1,pc2,wps,num_pcs,path):
    

    SIDE_RANGE=(-50,50)
    FWD_RANGE=(0,100)
    RES=0.15625/2
    
    npcs1=pc1.size()[2]
    npcs2=pc2.size()[2]
    
    pc_1=pc1[0].cpu().numpy()
    pc_2=pc2[0].cpu().numpy()
    wp_1=wps[0].cpu().detach().numpy()
    
    radar_ext = np.array([0.06, -0.2, 0.7,-3.5, 2, 180])
    ego_to_radar = get_matrix_from_ext(radar_ext)
    pc_1 = transform_to_ego(pc_1,ego_to_radar)
    pc_2 = transform_to_ego(pc_2,ego_to_radar)
    wp_1 = transform_to_ego(wp_1,ego_to_radar)

    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8)+255
    
    x_img_1 = np.floor((pc_1[0])/RES).astype(int)
    y_img_1 = np.floor(-(pc_1[1] + SIDE_RANGE[0])/RES).astype(int)
    for i in range(npcs1):
        im=cv2.circle(im,(x_img_1[i],y_img_1[i]),2,(0,0,0),2)
    
    x_img_2 = np.floor((pc_2[0])/RES).astype(int)
    y_img_2 = np.floor(- (pc_2[1] + SIDE_RANGE[0])/RES).astype(int)
    for j in range(npcs2):
        im=cv2.circle(im,(x_img_2[j],y_img_2[j]),2,(255,0,0),2)
    
    x_img_w = np.floor((wp_1[0])/RES).astype(int)
    y_img_w = np.floor(-(wp_1[1] + SIDE_RANGE[0])/RES).astype(int)
    x_img_w[x_img_w>(x_max-1)]=x_max-1
    y_img_w[y_img_w>(y_max-1)]=y_max-1
    x_img_w[x_img_w<0]=0
    y_img_w[y_img_w<0]=0
    
    for i in range(npcs1):
        im=cv2.line(im, (x_img_1[i],y_img_1[i]), (x_img_w[i],y_img_w[i]), (34,139,34), 1)
    
    path_im=path+'/'+'{}.png'.format(num_pcs)
    im=cv2.putText(im, 'PC1', (600,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    im=cv2.putText(im, 'PC2', (600,75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    im=cv2.putText(im, 'SF', (600,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (34,139,34), 2, cv2.LINE_AA)
    
    cv2.imwrite(path_im, im) 