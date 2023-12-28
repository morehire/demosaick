import torch


def getPSNR(groundTruth, recovered, peak=1):
    if groundTruth.is_cuda:
        groundTruth = groundTruth.detach().cpu()
    if recovered.is_cuda:
        recovered = recovered.detach().cpu()

    diff = groundTruth-recovered
    sqrd_error = torch.pow(diff, 2)
    mse = sqrd_error.mean()
    psnr = 10 * torch.log10((peak ** 2) / mse)
    return psnr


def getSAM(groundTruth, recovered):
    if groundTruth.is_cuda:
        groundTruth = groundTruth.detach().cpu()
    if recovered.is_cuda:
        recovered = recovered.detach().cpu()

    norm_groundTruth = batchNormalizeL2(groundTruth)
    norm_recovered = batchNormalizeL2(recovered)
    angles = torch.sum(norm_groundTruth * norm_recovered, dim=1)
    sams = torch.arccos(angles)
    return sams.mean()


def batchNormalizeL2(image, epsilon=1e-6):
    norm = torch.norm(image, p=2, dim=1, keepdim=True)
    mask = torch.abs(norm) < epsilon
    smoothed_norm = torch.where(mask, norm + epsilon, norm)
    norm_image = image / smoothed_norm
    return norm_image


if __name__ == "__main__":
    pass
