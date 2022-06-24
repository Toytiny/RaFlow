
import torch
from utils import *
from utils.model_utils import *
import torch.nn.functional as F


def computeSoftChamfer(pc1, pc1_warp, pc2, zeta=0.005):
    
    '''
    pc1: B 3 N
    pc2: B 3 N
    pc1_warp: B 3 N

    '''
    pc1 = pc1.permute(0, 2, 1)
    pc1_warp = pc1_warp.permute(0,2,1)
    pc2 = pc2.permute(0, 2, 1)
    npoints = pc1.size(1)
    batch_size = pc1.size(0)
    sqrdist12 = square_distance(pc1, pc2) # B N M

    ## use the kernel density estimation to obatin perpoint density
    dens12 = compute_density_loss(pc1, pc2, 1)
    dens21 = compute_density_loss(pc2, pc1, 1)

    mask1 = (dens12>zeta).type(torch.int32)
    mask2 = (dens21>zeta).type(torch.int32)

    sqrdist12w = square_distance(pc1_warp, pc2) # B N M

    dist1_w, _ = torch.topk(sqrdist12w, 1, dim = -1, largest=False, sorted=False)
    dist2_w, _ = torch.topk(sqrdist12w, 1, dim = 1, largest=False, sorted=False)
    dist1_w = dist1_w.squeeze(2)
    dist2_w = dist2_w.squeeze(1)
    
    dist1_w = F.relu(dist1_w-0.01)
    dist2_w = F.relu(dist2_w-0.01)
    
    dist1_w = dist1_w * mask1 
    dist2_w = dist2_w * mask2 

    
    return dist1_w, dist2_w


def computeWeightedSmooth(pc1, pred_flow, alpha=0.5):
    
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''
    
    B = pc1.size()[0] 
    N = pc1.size()[2]
    num_nb = 8
    pc1 = pc1.permute(0, 2, 1)
    npoints = pc1.size(1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    ## compute the neighbour distances in the point cloud
    dists, kidx = torch.topk(sqrdist, num_nb+1, dim = -1, largest=False, sorted=True)
    dists = dists[:,:,1:]
    kidx = kidx[:,:,1:]
    dists = torch.maximum(dists,torch.zeros(dists.size()).cuda())
    ## compute the weights according to the distances
    weights = torch.softmax(torch.exp(-dists/alpha).view(B,N*num_nb),dim=1)
    weights = weights.view(B,N,num_nb)
  
    grouped_flow = index_points_group(pred_flow, kidx) 
    diff_flow = (npoints*weights*torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3)).sum(dim = 2) 
   
    return diff_flow


    
def computeloss(pc1, pc2, agg_f, vel1,interval, args):
    
    f_smoothness = 1.0
    f_velo = 1.0
    f_chamfer = 1.0
    
    pc1_warp_a = pc1+agg_f
   
    N = pc1.size()[2]
    
    # chamfer 
    dist1_a, dist2_a = computeSoftChamfer(pc1, pc1_warp_a, pc2)
    chamferLoss =  torch.mean(dist1_a) + torch.mean(dist2_a)
    
    # smoothness
    diff_flow_a = computeWeightedSmooth(pc1, agg_f)
    smoothnessLoss =  torch.mean(diff_flow_a) 
    
    # velocity 
    ## the projection of the estimated flow on radical direction
    pred_fr_a=torch.sum(agg_f*pc1,dim=1)/(torch.norm(pc1,dim=1))
    diff_vel_a=torch.abs(vel1*interval.unsqueeze(1)-pred_fr_a)
    veloLoss= torch.mean(diff_vel_a) 
    
    
    total_loss = f_smoothness * smoothnessLoss + f_chamfer * chamferLoss + f_velo*veloLoss
    
    items={
        'Loss': total_loss.item(),
        'smoothnessLoss': smoothnessLoss.item(),
        'chamferLoss': chamferLoss.item(),
        'veloLoss': veloLoss.item(),
        }
    
    return total_loss, items
