import torch.nn as nn
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.model_utils import *
from utils import *


class RaFlow(nn.Module):
    def __init__(self,args):
        super(RaFlow,self).__init__()
        self.rigid_thres = 0.15
        self.rigid_pcs = 0.25
       
        self.npoints = args.num_points
        ## multi-scale set feature abstraction 
        # different scale share the same mlps hyper-parameters
        sa_radius = [2.0, 4.0, 8.0, 16.0]
        sa_nsamples = [4, 8, 16, 32]
        sa_mlps = [32, 32, 64]
        sa_mlp2s = [64,64,64]
        num_sas = len(sa_radius)
        self.mssa = MultiScalePointNet(sa_radius, sa_nsamples, in_channel=3, \
                                         mlp = sa_mlps, mlp2 = sa_mlp2s)
            
        ## feature correlation layer (cost volumn)
        fc_inch = num_sas*sa_mlp2s[-1]*2  
        fc_mlps = [fc_inch,fc_inch,fc_inch]
        self.fc_layer = PointConvFlow(8, in_channel=fc_inch*2+3, mlp=fc_mlps)
        
        ## multi-scale flow embeddings propogation
        # different scale share the same mlps hyper-parameters
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_nsamples = [4, 8, 16, 32]
        ep_inch = fc_inch * 2 + 3
        ep_mlps = [fc_inch, int(fc_inch/2), int(fc_inch/8)]
        ep_mlp2s = [int(fc_inch/8), int(fc_inch/8), int(fc_inch/8)]
        num_eps = len(sa_radius)
        self.msep = MultiScalePointNet(ep_radius, ep_nsamples, in_channel=ep_inch, \
                                         mlp = ep_mlps, mlp2 = ep_mlp2s)
        
        ## scene flow predictor
        sf_inch = num_eps * ep_mlp2s[-1]*2
        sf_mlps = [int(sf_inch/2), int(sf_inch/4), int(sf_inch/8)]
        self.fp = FlowPredictor(in_channel=sf_inch, mlp=sf_mlps)
    
    
    def rigid_to_flow(self,pc,trans):
        
        h_pc = torch.cat((pc,torch.ones((pc.size()[0],1,pc.size()[2])).cuda()),dim=1)
        sf = torch.matmul(trans,h_pc)[:,:3] - pc

        return sf
        
        
    def forward(self,pc1,pc2,feature1,feature2,interval):
        
        '''
        pc1: B 3 N
        pc2: B 3 N
        feature1: B 3 N
        feature2: B 3 N
        '''
        
        B = pc1.size()[0]
        N = pc1.size()[2]
        #cov_feature1 = torch.cat(((pc1[:,0]*pc1[:,1]).unsqueeze(2),\
        #    (pc1[:,1]*pc1[:,2]).unsqueeze(2),(pc1[:,2]*pc1[:,0]).unsqueeze(2)),dim=2).transpose(2,1).contiguous()
        #cov_feature2 = torch.cat(((pc2[:,0]*pc2[:,1]).unsqueeze(2),\
        #    (pc2[:,1]*pc2[:,2]).unsqueeze(2),(pc2[:,2]*pc2[:,0]).unsqueeze(2)),dim=2).transpose(2,1).contiguous()
        #feature1 = torch.cat((feature1,cov_feature1),dim=1)
        #feature2 = torch.cat((feature2,cov_feature2),dim=1)
        ## extract multi-scale local features for each point
        #feature1 = feature1[:,(0,1),:]
        #feature2 = feature2[:,(0,1),:]
        pc1_features = self.mssa(pc1,feature1)
        pc2_features = self.mssa(pc2,feature2)
        
        ## global features for each set
        gfeat_1 = torch.max(pc1_features,-1)[0].unsqueeze(2).expand(pc1_features.size()[0],pc1_features.size()[1],pc1.size()[2])
        gfeat_2 = torch.max(pc2_features,-1)[0].unsqueeze(2).expand(pc2_features.size()[0],pc2_features.size()[1],pc2.size()[2])
        
        ## concat local and global features
        pc1_features = torch.cat((pc1_features, gfeat_1),dim=1)
        pc2_features = torch.cat((pc2_features, gfeat_2),dim=1)
        
        ## associate data from two sets 
        cor_features = self.fc_layer(pc1, pc2, pc1_features, pc2_features)
        embeddings = torch.cat((feature1, pc1_features, cor_features),dim=1)
        
        ## multi-scale flow embeddings propogation
        prop_features = self.msep(pc1,embeddings)
        gfeat = torch.max(prop_features,-1)[0].unsqueeze(2).expand(prop_features.size()[0],prop_features.size()[1],pc1.size()[2])
        final_features = torch.cat((prop_features, gfeat),dim=1)
        
        ## initial scene flow prediction
        output = self.fp(final_features)
        
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
       
        return output, sf_agg, pre_trans, mask_s
    
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
     
