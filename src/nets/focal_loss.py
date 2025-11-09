import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha)
            else:
                self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # [N, C, H, W] => [N, C, HW]
            input = input.transpose(1, 2)  # [N, C, HW] => [N, HW, C]
            input = input.contiguous().view(-1, input.size(-1))  # [N, HW, C] => [NHW, C]
        target = target.view(-1)

        log_softmax = F.log_softmax(input, dim=-1)
        ce = F.nll_loss(log_softmax, target, weight=self.alpha, reduction='none')
        
        pred = log_softmax.exp()
        pred = pred.gather(1, target.unsqueeze(1)).squeeze(1)  # 获取对应真实类别的预测概率
        
        focal_weight = (1 - pred) ** self.gamma
        
        loss = focal_weight * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
