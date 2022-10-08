import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.model_utils import *
from utils import *


class RaFlow_VoD(nn.Module):

    '''
    The model implementation of RaFlow (for running on the view-of-delft dataset)
    Update: put all module componenets in this file and rename some of them
    
    '''
    
    def __init__(self,args):
        
        super(RaFlow_VoD,self).__init__()
        
        self.rigid_thres = 0.15
        self.rigid_pcs = 0.25
        self.npoints = args.num_points
        
        ## multi-scale set feature abstraction 
        sa_radius = [2.0, 4.0, 8.0, 16.0]
        sa_nsamples = [4, 8, 16, 32]
        sa_mlps = [32, 32, 64]
        sa_mlp2s = [64,64,64]
        num_sas = len(sa_radius)
        self.mse_layer = MultiScaleEncoder(sa_radius, sa_nsamples, in_channel=3, \
                                         mlp = sa_mlps, mlp2 = sa_mlp2s)
            
        ## feature correlation layer (cost volumn)
        fc_inch = num_sas*sa_mlp2s[-1]*2  
        fc_mlps = [fc_inch,fc_inch,fc_inch]
        self.fc_layer = FeatureCorrelator(8, in_channel=fc_inch*2+3, mlp=fc_mlps)
        
        ## flow decoder layer (output coarse scene flow)
        self.fd_layer = FlowDecoder(fc_inch=fc_inch)
        
    
    def rigid_to_flow(self,pc,trans):
        
        h_pc = torch.cat((pc,torch.ones((pc.size()[0],1,pc.size()[2])).cuda()),dim=1)
        sf = torch.matmul(trans,h_pc)[:,:3] - pc

        return sf
        
    
    def ROFE_module(self,pc1,pc2,feature1,feature2):
        
        '''
        pc1: B 3 N
        pc2: B 3 N
        feature1: B 3 N
        feature2: B 3 N
        
        '''
        
        B = pc1.size()[0]
        N = pc1.size()[2]
        ## extract multi-scale local features for each point
        pc1_features = self.mse_layer(pc1,feature1)
        pc2_features = self.mse_layer(pc2,feature2)
        
        ## global features for each set
        gfeat_1 = torch.max(pc1_features,-1)[0].unsqueeze(2).expand(pc1_features.size()[0],pc1_features.size()[1],pc1.size()[2])
        gfeat_2 = torch.max(pc2_features,-1)[0].unsqueeze(2).expand(pc2_features.size()[0],pc2_features.size()[1],pc2.size()[2])
        
        ## concat local and global features
        pc1_features = torch.cat((pc1_features, gfeat_1),dim=1)
        pc2_features = torch.cat((pc2_features, gfeat_2),dim=1)
        
        ## associate data from two sets 
        cor_features = self.fc_layer(pc1, pc2, pc1_features, pc2_features)
        
        ## decoding scene flow from embeedings
        output = self.fd_layer(pc1, feature1, pc1_features, cor_features)
        
        return output 
    
    def SFR_module(self, output, pc1, feature1, interval):
        
        B = pc1.size()[0]
        N = pc1.size()[2]
        
        ## warped pc1 with scene flow estimation
        pc1_warp = pc1+output
        
        ## estimate rigid transformation using initial scene flow
        ## assume all points static
        mask = torch.ones((pc1.size()[0],pc1.size()[2])).cuda()
        trans = self.rigid_transform_torch(pc1, pc1_warp, mask)
        # from transformation to rigid scene flow
        sf_rg = self.rigid_to_flow(pc1,trans)
        
        # mask for static points approximation by radial projection threshold
        vel_1 = feature1[:,0]
        sf_proj=torch.sum(sf_rg*pc1,dim=1)/(torch.norm(pc1,dim=1))
        residual=(vel_1*interval.unsqueeze(1)-sf_proj)
        mask_s = (abs(residual/vel_1) < self.rigid_thres) 
        
        # when enough points are inliers of a rigid transformation
        # use the rigid transformation to replace individual flow vectors of them
        pre_trans = torch.zeros(trans.size()).cuda()
        sf_agg = torch.zeros(output.size()).cuda()
        for b in range(B):
            if (mask_s[b].sum()/N)>self.rigid_pcs:
                pre_trans[b] = self.rigid_transform_torch(pc1[b].unsqueeze(0), \
                                    pc1_warp[b].unsqueeze(0), mask_s[b].unsqueeze(0))
                # from transformation to rigid scene flow
                sf_agg[b] = self.rigid_to_flow(pc1[b].unsqueeze(0),pre_trans[b].unsqueeze(0))
                sf_agg[b,:,torch.logical_not(mask_s[b])]=output[b,:,torch.logical_not(mask_s[b])]
            else:
                pre_trans[b] = trans[b]
                sf_agg[b] = output[b]
                
        return sf_agg, pre_trans, mask_s
    
    def rigid_transform_torch(self, A, B, M):
    
        assert A.size() == B.size()
    
        batch_size, num_rows, num_cols = A.size()
       
        ## mask to 0/1 weights for motive/static points
        W=M.type(torch.bool).unsqueeze(2)

        # find mean column wise
        centroid_A = torch.mean(A.transpose(2,1).contiguous()*W, axis=1)
        centroid_B = torch.mean(B.transpose(2,1).contiguous()*W, axis=1)
    
        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(batch_size,num_rows,1)
        centroid_B = centroid_B.reshape(batch_size,num_rows,1)
    
        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B
    
        H = torch.matmul(Am, Bm.transpose(2,1).contiguous()*W)

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
                 
    def forward(self,pc1,pc2,feature1,feature2,interval):
        

        output = self.ROFE_module(pc1,pc2,feature1,feature2)
        sf_agg, pre_trans, mask_s = self.SFR_module(output, pc1, feature1, interval)
        
   
        return output, sf_agg, pre_trans, mask_s
    

class MultiScaleEncoder(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(MultiScaleEncoder, self).__init__()

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
        

class FeatureCorrelator(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = False, use_leaky = True):
        super(FeatureCorrelator, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1) # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        return patch_to_patch_cost    


 
class FlowDecoder(nn.Module):
    def __init__(self, fc_inch):
        super(FlowDecoder, self).__init__()
        ## multi-scale flow embeddings propogation
        # different scale share the same mlps hyper-parameters
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(ep_radius)
        self.mse = MultiScaleEncoder(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        ## scene flow predictor
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowPredictor(in_channel=sf_inch, mlp=sf_mlps)
        
    def forward(self, pc1, feature1, pc1_features, cor_features):
        
        embeddings = torch.cat((feature1, pc1_features, cor_features),dim=1)
        ## multi-scale flow embeddings propogation
        prop_features = self.mse(pc1,embeddings)
        gfeat = torch.max(prop_features,-1)[0].unsqueeze(2).expand(prop_features.size()[0],prop_features.size()[1],pc1.size()[2])
        final_features = torch.cat((prop_features, gfeat),dim=1)
        
        ## initial scene flow prediction
        output = self.fp(final_features)
        
        return output