import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        # import pdb; pdb.set_trace()
        pred = torch.unsqueeze(pred, 1)
        l = ((pred - gt)**2)
        l = l.mean(dim=2).mean(dim=1)
        return l ## l of dim bsize