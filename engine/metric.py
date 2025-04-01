import numpy as np


def hist_info(n_cl, pred, gt, excluded_labels=np.array([])):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl) & (pred < n_cl) & (
        ~np.isin(gt, excluded_labels))
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                                  minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct


def compute_score(hist, correct, labeled):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])  # useless for NYUDv2

    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)

    pixel_acc = correct / labeled

    return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = np.array([])
    k = (a > 3) & (a < 9) & (~np.isin(a, b))
    print(k)
    print(np.sum(k))
