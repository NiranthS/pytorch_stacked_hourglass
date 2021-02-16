import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
from task.loss import HeatmapLoss
import torch.nn.functional as F

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, inp_dim)
        )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

        self.conv_1 = nn.Conv2d(inp_dim, 128, 3)
        self.conv_2 = nn.Conv2d(128, 64, 3)

        self.dense1 = nn.Linear(64*60*60, 1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.dense3 = nn.Linear(1024, 82)
        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.5)
        
        

    def forward(self, imgs):
        ## our posenet
        x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        # import pdb; pdb.set_trace()
        features_ds = F.relu(self.conv_1(feature))
        features_ds = F.relu(self.conv_2(features_ds))
        
        features_ds = features_ds.view(-1, 64*60*60)
        dense = F.relu(self.dense1(features_ds))
        dense = self.do1(dense)
        dense = F.relu(self.dense2(dense))
        dense = self.do2(dense)
        dense = self.dense3(dense)
        combined_hm_preds.append(dense)
        # return torch.stack(combined_hm_preds, 1)
        return dense

    def calc_loss(self, combined_hm_preds, heatmaps):
        # for i in range(self.nstack):
        #     combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        # combined_loss = torch.stack(combined_loss, dim=1)
        combined_loss = self.heatmapLoss(combined_hm_preds, heatmaps)
        return combined_loss
