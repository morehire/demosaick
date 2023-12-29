import torch
import torch.nn as nn

class MARELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(MARELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, groundTruth, recovered):
        mask = torch.abs(groundTruth) < self.epsilon
        smoothed_groundTruth = torch.where(mask, groundTruth + self.epsilon, groundTruth)
        difference = torch.abs((groundTruth - recovered) / smoothed_groundTruth)
        mare = torch.mean(difference)
        return mare

class FFTLoss(nn.Module):
    def __init__(self,epsilon = 1e-6):
        super(FFTLoss, self).__init__()
        self.epsilon = epsilon
        self.L1Loss = nn.L1Loss()
    def forward(self,groundTruth,recovered):
        gt_fft = torch.abs(torch.fft.fftn(groundTruth,dim=(-2,-1)))
        recover_fft = torch.abs(torch.fft.fftn(recovered,dim=(-2,-1)))
        return self.L1Loss(gt_fft,recover_fft)

class MixLoss_fft(nn.Module):
    def __init__(self, losses, weights):
        super(MixLoss_fft, self).__init__()
        self.losses = [getattr(nn, loss)() if loss != 'MARELoss' else MARELoss() for loss in losses]
        self.fft_loss = FFTLoss()
        self.weights = weights
        #self.scale = scale

    def forward(self, groundTruth, recovered):
        # penalty_strength = 0.1
        # penalty = penalty_strength * torch.max(torch.tensor(0.0), scale - 1.0) + penalty_strength * torch.max(torch.tensor(0.0), 2.0 - scale)
        loss = sum(w * loss_fn(groundTruth, recovered) for w, loss_fn in zip(self.weights, self.losses))
        fft_loss = self.fft_loss(groundTruth,recovered)
        loss += 0.2*fft_loss
        #loss+=penalty
        return loss

class PPILoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,groundTruth,recovered):
        gt_ppi = self.get_PPI(groundTruth)
        re_ppi = self.get_PPI(recovered)
        difference = torch.abs(gt_ppi-re_ppi)
        ppi_loss = torch.mean(difference)
        return ppi_loss
    def get_PPI(self, img):
        #按通道维计算ppi
        ppi = torch.mean(img,dim=1)
        return ppi

class MixLoss_ppi(nn.Module):
    def __init__(self, losses, weights):
        super(MixLoss_ppi, self).__init__()
        self.losses = [getattr(nn, loss)() if loss != 'MARELoss' else MARELoss() for loss in losses]
        self.weights = weights
        self.ppi_loss = PPILoss()

    def forward(self, groundTruth, recovered):
        loss = sum(w * loss_fn(groundTruth, recovered) for w, loss_fn in zip(self.weights, self.losses))
        ppi_loss = self.ppi_loss(groundTruth,recovered)
        return loss + ppi_loss

class MixLoss(nn.Module):
    def __init__(self, losses, weights):
        super(MixLoss, self).__init__()
        self.losses = [getattr(nn, loss)() if loss != 'MARELoss' else MARELoss() for loss in losses]
        self.weights = weights
        #self.scale = scale

    def forward(self, groundTruth, recovered):
        # penalty_strength = 0.1
        # penalty = penalty_strength * torch.max(torch.tensor(0.0), scale - 1.0) + penalty_strength * torch.max(torch.tensor(0.0), 2.0 - scale)
        loss = sum(w * loss_fn(groundTruth, recovered) for w, loss_fn in zip(self.weights, self.losses))
        #loss+=penalty
        return loss

class MixLoss_re(nn.Module):
    """
    带有对scale参数正则化项的损失
    """
    def __init__(self, losses, weights):
        super(MixLoss_re, self).__init__()
        self.losses = [getattr(nn, loss)() if loss != 'MARELoss' else MARELoss() for loss in losses]
        self.weights = weights
        #self.scale = scale

    def forward(self, groundTruth, recovered,scale,beta):
        delta = 0.01
        # 正则化项 避免参数超出范围
        res = torch.max((scale - 1.00 - delta) / (-delta), (scale - 1.1 + delta) / delta)
        range_loss = torch.mean(torch.max(res, torch.zeros_like(res)))
        loss = sum(w * loss_fn(groundTruth, recovered) for w, loss_fn in zip(self.weights, self.losses))

        loss = loss+beta.item()*range_loss
        return loss

if __name__ == "__main__":
    groundTruth = torch.tensor([3.0, 5.0, 7.0], requires_grad=True)
    recovered = torch.tensor([2.5, 5.2, 6.8], requires_grad=True)
    losses = ['L1Loss', 'MSELoss', 'MARELoss']
    weights = [0.3, 0.5, 0.2]

    mix_loss_fn = MixLoss(losses, weights)
    mixed_loss = mix_loss_fn(groundTruth, recovered)

    print("Mixed Loss:", mixed_loss.item())
