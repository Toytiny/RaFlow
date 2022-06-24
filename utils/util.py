import torch
from time import time
import numpy as np


# Batched index_select
def batched_index_select(t, dim, inds):
    dummy = inds.expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


def rigid_transform_torch(A, B):

    assert A.size() == B.size()

    batch_size, num_rows, num_cols = A.size()
   
    # find mean column wise
    centroid_A = torch.mean(A.transpose(2,1).contiguous(), axis=1)
    centroid_B = torch.mean(B.transpose(2,1).contiguous(), axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(batch_size,num_rows,1)
    centroid_B = centroid_B.reshape(batch_size,num_rows,1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = torch.matmul(Am, Bm.transpose(2,1).contiguous())

    # find rotation
    U, S, V = torch.svd(H)
    Z = torch.matmul(V,U.transpose(2,1).contiguous())
    # special reflection case
    d= (torch.linalg.det(Z) < 0).type(torch.int8)
    # -1/1 
    d=d*2-1
    Vc = V.clone()
    Vc[:,2,:]*=-d.view(batch_size,1)
    R = torch.matmul(Vc,U.transpose(2,1).contiguous())
   
    t = torch.matmul(-R, centroid_A)+centroid_B
    
    Trans=torch.cat((torch.cat((R,t),axis=2), torch.tensor([0,0,0,1]).repeat(batch_size,1).cuda().view(batch_size,1,4)),axis=1)

    return Trans

def rigid_transform_3D(A, B):
    
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    
    
    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)
 

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    
    Trans=np.concatenate((np.concatenate((R,t),axis=1), np.expand_dims(np.array([0,0,0,1]),0)),axis=0)

    return Trans


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.maximum(dist,torch.zeros(dist.size()).cuda())
    return dist

def compute_density_loss(xyz1, xyz2, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz1.shape
    sqrdists = square_distance(xyz1, xyz2)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density






