import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from lib import pointnet2_utils as pointutils

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


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

class MultiScalePointNet(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(MultiScalePointNet, self).__init__()

        self.ms_ls = nn.ModuleList()
        num_sas = len(radius)
        for l in range(num_sas):
            self.ms_ls.append(PointLocalFeature(radius[l], \
                                    nsample[l],in_channel=in_channel, mlp=mlp, mlp2=mlp2))
                
    def forward(self, xyz, features):
        
        new_features = torch.zeros(0).cuda()
        
        for i, sa in enumerate(self.ms_ls):
            new_features = torch.cat((new_features,sa(xyz,features)),dim=1)
            
        return new_features
        
    
class PointLocalFeature(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(PointLocalFeature, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
        last_channel = mlp[-1]
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
        self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
  
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        new_points = self.queryandgroup(xyz_t, xyz_t, points) 
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0].unsqueeze(2)
        
        for i, conv in enumerate(self.mlp2_convs):
            bn = self.mlp2_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        new_points = new_points.squeeze(2)
        
        return new_points
    
    
class FlowPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FlowPredictor, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
            
        self.conv2 = nn.Conv2d(mlp[-1], 3, 1, bias=False)
        
    def forward(self, feat):
 
        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)
        
        output = self.conv2(feat)
        
        return output.squeeze(3)

    
def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density
        
