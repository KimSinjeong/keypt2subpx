import argparse
import torch
from torch.nn import functional as F
import numpy as np
import os
import pickle
from tqdm import tqdm

from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.utils.tensor import batch_to_device
from gluefactory.utils.patches import extract_patches

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# parse command line arguments
parser = argparse.ArgumentParser(
	description='Preprocess MegaDepth dataset with XFeat.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--variant', '-v', default='train', choices=['train', 'val', 'test'],
	help='Defines subfolders of the dataset ot use (split according to "Glue Factory", https://github.com/cvg/glue-factory).')

parser.add_argument('--topk', '-tk', type=int, default=4096, 
	help='number of features per image')

parser.add_argument('--patch_radius', '-ps', type=int, default=5, 
	help='patch radius')

parser.add_argument('--dataconf', '-dc', type=str, default='./configs/megadepth_dataconf_b1_grey.yaml',
	help='data configuration file path')

opt = parser.parse_args()

# setup dataset & dataloader
dataconf = OmegaConf.load(opt.dataconf)
dataset = get_dataset(dataconf.name)(dataconf)
dataloader = dataset.get_data_loader(opt.variant, distributed=False, shuffle=False)
print(f'Using dataset: {opt.dataconf} {opt.variant}', flush=True)

# output folder that stores pre-calculated correspondence vectors as PyTorch tensors
out_dir = 'data/megadepth/' + opt.variant + '_xfeat/'
if not os.path.isdir(out_dir): os.makedirs(out_dir)

# setup detector
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = opt.topk).cuda()
xfeat.eval()

# The matcher part is borrowed and modified from GlueFactory (https://github.com/cvg/glue-factory)
# Nearest neighbor matcher for normalized descriptors.
# Optionally apply the mutual check and threshold the distance or ratio.
@torch.no_grad()
def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2) * dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    return matches


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    inds1 = torch.arange(m1.shape[-1], device=m1.device)
    loop0 = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    loop1 = torch.gather(m0, -1, torch.where(m1 > -1, m1, m1.new_tensor(0)))
    m0_new = torch.where((m0 > -1) & (inds0 == loop0), m0, m0.new_tensor(-1))
    m1_new = torch.where((m1 > -1) & (inds1 == loop1), m1, m1.new_tensor(-1))
    return m0_new, m1_new

ratio_thresh = None
distance_thresh = None

def nearest_neighbor_match(descriptors0, descriptors1):
    sim = torch.einsum("bnd,bmd->bnm", descriptors0, descriptors1)
    matches0 = find_nn(sim, ratio_thresh, distance_thresh)
    matches1 = find_nn(
        sim.transpose(1, 2), ratio_thresh, distance_thresh
    )
    matches0, matches1 = mutual_check(matches0, matches1)
    b, m, n = sim.shape
    la = sim.new_zeros(b, m + 1, n + 1)
    la[:, :-1, :-1] = F.log_softmax(sim, -1) + F.log_softmax(sim, -2)
    mscores0 = (matches0 > -1).float()
    mscores1 = (matches1 > -1).float()
    return {
        "matches0": matches0,
        "matches1": matches1,
        "matching_scores0": mscores0,
        "matching_scores1": mscores1,
        "similarity": sim,
        "log_assignment": la,
    }
# Borrowed part ends

with torch.no_grad():
    for i, data in enumerate(tqdm(dataloader)):
        scene = data['view0']['scene'][0]
        assert data['view0']['scene'][0] == data['view1']['scene'][0], "Scenes' names are not the same"
        img1_name = data['view0']['name'][0].split('.')[0]
        img2_name = data['view1']['name'][0].split('.')[0]

        img1_shape = data['view0']['image_size'][0].numpy() # W, H
        img1_shape = np.array([img1_shape[1], img1_shape[0], 3]) # H, W, 3
        img2_shape = data['view1']['image_size'][0].numpy() # W, H
        img2_shape = np.array([img2_shape[1], img2_shape[0], 3]) # H, W, 3

        tqdm.write("Processing pair %5d of %5d. (%30s, %30s)" % (i, len(dataloader), img1_name, img2_name))

        gpu_data = batch_to_device(data, 'cuda:0', non_blocking=True)

        output = xfeat.detectAndCompute(gpu_data['view0']['image'][:,:,:img1_shape[0],:img1_shape[1]], top_k=4096)[0]
        pts1, score1, desc1 = output['keypoints'], output['scores'], output['descriptors']
        output = xfeat.detectAndCompute(gpu_data['view1']['image'][:,:,:img2_shape[0],:img2_shape[1]], top_k=4096)[0]
        pts2, score2, desc2 = output['keypoints'], output['scores'], output['descriptors']

        pred = nearest_neighbor_match(desc1.unsqueeze(0), desc2.unsqueeze(0))

        matches1 = pred['matches0'][0]
        mask1 = matches1 != -1
        pts1 = pts1[mask1, :2]
        pts2 = pts2[matches1[mask1], :2]

        desc1 = desc1[mask1] # N * 256
        desc2 = desc2[matches1[mask1]] # N * 256

        K1 = data['view0']['camera'].calibration_matrix()[0] # B * 3 * 3
        K2 = data['view1']['camera'].calibration_matrix()[0] # B * 3 * 3

        GT_R_Rel = data['T_0to1'].R[0]
        GT_t_Rel = data['T_0to1'].t[0].unsqueeze(-1)

        ratios = torch.zeros(pts1.shape[0]) # N

        # Patches start
        bias = torch.tensor([[opt.patch_radius]*2], device=pts1.device)
        pad_amount = opt.patch_radius

        assert (data['view0']['image'][0] < (1. + 1e-5)).all() and (data['view0']['image'][0] > -(1. + 1e-5)).all(), "Image 0 out of range"

        if data['view0']['image'].shape[1] == 3:  # RGB
            scale = data['view0']['image'].new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            data['view0']['image'] = (data['view0']['image'] * scale).sum(1, keepdim=True)

            scale = data['view1']['image'].new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            data['view1']['image'] = (data['view1']['image'] * scale).sum(1, keepdim=True)

        img1_padded = torch.nn.functional.pad(data['view0']['image'][0][:, :img1_shape[0], :img1_shape[1]], [pad_amount]*4, mode='constant', value=0.)
        patch1 = extract_patches(img1_padded, (pts1 - bias + pad_amount).to(device=img1_padded.device, dtype=torch.int32), 2*opt.patch_radius+1)[0]

        assert (data['view1']['image'][0] < (1. + 1e-5)).all() and (data['view1']['image'][0] > -(1. + 1e-5)).all(), "Image 1 out of range"
        img2_padded = torch.nn.functional.pad(data['view1']['image'][0][:, :img2_shape[0], :img2_shape[1]], [pad_amount]*4, mode='constant', value=0.)
        patch2 = extract_patches(img2_padded, (pts2 - bias + pad_amount).to(device=img2_padded.device, dtype=torch.int32), 2*opt.patch_radius+1)[0]
        # Patches end

        with open(out_dir + f'pair-{scene}-{img1_name}-{img2_name}.pkl', 'wb') as f:
            pickle.dump([
                pts1.cpu().numpy().astype(np.float32), # N, 2
                pts2.cpu().numpy().astype(np.float32), # N, 2
                ratios.cpu().numpy().astype(np.float32), # N
                img1_shape, # H, W, 3
                img2_shape, # H, W, 3
                K1.numpy().astype(np.float32), # 3, 3
                K2.numpy().astype(np.float32), # 3, 3
                GT_R_Rel.numpy().astype(np.float32), # 3, 3
                GT_t_Rel.numpy().astype(np.float32), # 3, 1
                patch1.cpu().numpy().astype(np.float32), # N, C, 2*patch_radius+1, 2*patch_radius+1
                patch2.cpu().numpy().astype(np.float32), # N, C, 2*patch_radius+1, 2*patch_radius+1
                desc1.cpu().numpy().astype(np.float32), # N, D
                desc2.cpu().numpy().astype(np.float32),  # N, D
            ], f)

# pts1:  (N, 2)
# pts2:  (N, 2)
# ratios:  (N,)
# img1:  (H, W, 3)
# img2:  (H, W, 3)
# K1:  (3, 3)
# K2:  (3, 3)
# GT_R_Rel:  (3, 3)
# GT_t_Rel:  (3, 1)