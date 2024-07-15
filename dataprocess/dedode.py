import argparse
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import pickle
from tqdm import tqdm

from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.utils.tensor import batch_to_device
from gluefactory.utils.patches import extract_patches

from kornia.feature import DeDoDe

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# parse command line arguments
parser = argparse.ArgumentParser(
	description='Preprocess MegaDepth dataset with Kornia DeDoDe.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--variant', '-v', default='train', choices=['train', 'val', 'test'],
	help='Defines subfolders of the dataset ot use (split according to "Glue Factory", https://github.com/cvg/glue-factory).')

parser.add_argument('--nfeatures', '-nf', type=int, default=10_000, 
	help='number of features per image')

parser.add_argument('--patch_radius', '-ps', type=int, default=5, 
	help='patch radius')

parser.add_argument('--dataconf', '-dc', type=str, default='./configs/megadepth_dataconf_b1.yaml',
	help='data configuration file path')

opt = parser.parse_args()

# setup dataset & dataloader
dataconf = OmegaConf.load(opt.dataconf)
dataset = get_dataset(dataconf.name)(dataconf)
dataloader = dataset.get_data_loader(opt.variant, distributed=False, shuffle=False)
print(f'Using dataset: {opt.dataconf} {opt.variant}', flush=True)

# output folder that stores pre-calculated correspondence vectors as PyTorch tensors
out_dir = 'data/megadepth/' + opt.variant + '_dedode/'
if not os.path.isdir(out_dir): os.makedirs(out_dir)

# This part is borrowed and modified from DeDoDe's Official Dual Softmax Matcher implementation (https://github.com/Parskatt/DeDoDe)
def dual_softmax_matcher(desc_A: tuple['B','N','C'], desc_B: tuple['B','M','C'], inv_temperature = 1, normalize = False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    B, N, C = desc_A.shape
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim = -2) * corr.softmax(dim= -1)
    return P

def to_pixel_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return flow

def to_normalized_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                2 * (flow[..., 0]) / w1 - 1,
                2 * (flow[..., 1]) / h1 - 1,
            ),
            axis=-1,
        )
    )
    return flow

class DualSoftMaxMatcher(nn.Module):        
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = dual_softmax_matcher(descriptions_A, descriptions_B, 
                                 normalize = normalize, inv_temperature=inv_temp,
                                 )
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))
        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]
        return matches_A, matches_B, inds

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)
# Part borrowed from DeDoDe ends

# setup detector
dedode = DeDoDe.from_pretrained(detector_weights="L-upright", descriptor_weights="G-upright").cuda()
dedode.eval()
matcher = DualSoftMaxMatcher()

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

        keypoints0, P_0 = dedode.detect(gpu_data['view0']['image'][:,:,:img1_shape[0],:img1_shape[1]], n = opt.nfeatures)
        keypoints1, P_1 = dedode.detect(gpu_data['view1']['image'][:,:,:img2_shape[0],:img2_shape[1]], n = opt.nfeatures)

        resized_0 = torchvision.transforms.Resize((int(np.ceil(img1_shape[0]//14)*14), int(np.ceil(img1_shape[1]//14)*14)))(gpu_data['view0']['image'][:,:,:img1_shape[0],:img1_shape[1]])
        resized_1 = torchvision.transforms.Resize((int(np.ceil(img2_shape[0]//14)*14), int(np.ceil(img2_shape[1]//14)*14)))(gpu_data['view1']['image'][:,:,:img2_shape[0],:img2_shape[1]])

        desc1 = dedode.describe(resized_0, keypoints0)
        desc2 = dedode.describe(resized_1, keypoints1)

        matches_0, matches_1, inds = matcher.match(keypoints0, desc1, keypoints1, desc2,
                P_A = P_0, P_B = P_1, normalize = True, inv_temp=20, threshold = 0.1)

        batch_inds = inds[:,0]
        desc1 = desc1[batch_inds, inds[:,1]]
        desc2 = desc2[batch_inds, inds[:,2]]
        
        pts1, pts2 = matcher.to_pixel_coords(matches_0, matches_1, img1_shape[0], img1_shape[1], img2_shape[0], img2_shape[1])

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