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


# ---------------------------------------------------------------------------
# GOOSE competition metrics (CodaBench ICRA 2026)
# ---------------------------------------------------------------------------
# 8 classes excluded from evaluation (absent or <10 images in test set):
#   0=undefined, 7=bikeway, 9=pedestrian_crossing, 35=on_rails,
#   44=tunnel, 56=outlier, 61=pipe, 63=military_vehicle
# This leaves 56 fine classes mapped to 11 coarse superclasses.

GOOSE_EXCLUDED_LABELS = {0, 7, 9, 35, 44, 56, 61, 63}

GOOSE_FINE_TO_COARSE = {
    # Animal
    33: 0,   # animal
    # Construction
    43: 1,   # bridge
    38: 1,   # building
    58: 1,   # container
    29: 1,   # debris
    41: 1,   # fence
    42: 1,   # guard_rail
    39: 1,   # wall
    55: 1,   # wire
    # Human
    14: 2,   # person
    32: 2,   # rider
    # Object
    4:  3,   # obstacle
    45: 3,   # pole
    6:  3,   # street_light
    40: 3,   # rock
    60: 3,   # barrel
    # Road
    22: 4,   # curb
    26: 4,   # rail_track
    11: 4,   # road_marking
    21: 4,   # sidewalk
    # Sign
    48: 5,   # barrier_tape
    47: 5,   # misc_sign
    1:  5,   # traffic_cone
    19: 5,   # traffic_light
    46: 5,   # traffic_sign
    10: 5,   # road_block
    25: 5,   # boom_barrier
    # Sky
    53: 6,   # sky
    # Terrain
    23: 7,   # asphalt
    3:  7,   # cobble
    24: 7,   # gravel
    31: 7,   # soil
    2:  7,   # snow
    # Vegetation
    17: 8,   # bush
    30: 8,   # crops
    16: 8,   # forest
    59: 8,   # hedge
    51: 8,   # high_grass
    5:  8,   # leaves
    50: 8,   # low_grass
    18: 8,   # moss
    52: 8,   # scenery_vegetation
    27: 8,   # tree_crown
    28: 8,   # tree_trunk
    62: 8,   # tree_root
    # Vehicle
    8:  9,   # ego_vehicle
    13: 9,   # bicycle
    15: 9,   # bus
    12: 9,   # car
    36: 9,   # caravan
    57: 9,   # heavy_machinery
    49: 9,   # kick_scooter
    20: 9,   # motorcycle
    37: 9,   # trailer
    34: 9,   # truck
    # Water
    54: 10,  # water
}

GOOSE_COARSE_NAMES = [
    "Animal", "Construction", "Human", "Object", "Road",
    "Sign", "Sky", "Terrain", "Vegetation", "Vehicle", "Water",
]

# Fine class IDs that participate in evaluation (sorted)
GOOSE_FINE_IDS = sorted(GOOSE_FINE_TO_COARSE.keys())

NUM_GOOSE_COARSE = len(GOOSE_COARSE_NAMES)  # 11


def goose_compute_scores(hist_fine, num_classes=64):
    """Compute mIoU_fine, mIoU_coarse, mIoU_composite from a (num_classes x num_classes) confusion matrix.

    Only the 56 non-void classes participate in mIoU_fine.
    For mIoU_coarse the confusion matrix is collapsed to the 11 superclasses.
    """
    # --- mIoU_fine: per-class IoU over the 56 evaluated classes ---
    fine_ious = []
    for c in GOOSE_FINE_IDS:
        intersection = hist_fine[c, c]
        union = hist_fine[c, :].sum() + hist_fine[:, c].sum() - intersection
        if union == 0:
            continue  # class not present in gt or pred, skip
        fine_ious.append(intersection / union)

    mIoU_fine = np.mean(fine_ious) if fine_ious else 0.0

    # --- mIoU_coarse: collapse to 11 superclasses ---
    hist_coarse = np.zeros((NUM_GOOSE_COARSE, NUM_GOOSE_COARSE), dtype=np.float64)
    for src_fine, src_coarse in GOOSE_FINE_TO_COARSE.items():
        for dst_fine, dst_coarse in GOOSE_FINE_TO_COARSE.items():
            hist_coarse[src_coarse, dst_coarse] += hist_fine[src_fine, dst_fine]

    coarse_ious = []
    for c in range(NUM_GOOSE_COARSE):
        intersection = hist_coarse[c, c]
        union = hist_coarse[c, :].sum() + hist_coarse[:, c].sum() - intersection
        if union == 0:
            continue
        coarse_ious.append(intersection / union)

    mIoU_coarse = np.mean(coarse_ious) if coarse_ious else 0.0

    # --- mIoU_composite ---
    mIoU_composite = 0.5 * mIoU_fine + 0.5 * mIoU_coarse

    return mIoU_fine, mIoU_coarse, mIoU_composite, fine_ious, coarse_ious


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    b = np.array([])
    k = (a > 3) & (a < 9) & (~np.isin(a, b))
    print(k)
    print(np.sum(k))
