import argparse
import torch
import numpy as np
import os
import pickle
from tqdm import tqdm

from omegaconf import OmegaConf

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.utils.tensor import batch_to_device
from gluefactory.utils.patches import extract_patches

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# parse command line arguments
parser = argparse.ArgumentParser(
	description='Preprocess MegaDepth dataset with GlueFactory.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--variant', '-v', default='train', choices=['train', 'val', 'test'],
	help='Defines subfolders of the dataset ot use (split according to "Glue Factory", https://github.com/cvg/glue-factory).')

parser.add_argument('--nfeatures', '-nf', type=int, default=2048, 
	help='number of features per image, -1 does not restrict feature count')

parser.add_argument('--conf_thresh', '-ct', type=float, default=0.0, 
	help='confidence threshold')

parser.add_argument('--patch_radius', '-ps', type=int, default=5, 
	help='patch radius')

parser.add_argument('--dataconf', '-dc', type=str, default='./configs/megadepth_dataconf_b1.yaml',
	help='data configuration file path')

opt = parser.parse_args()

dataconf = OmegaConf.load(opt.dataconf)
dataset = get_dataset(dataconf.name)(dataconf)
dataloader = dataset.get_data_loader(opt.variant, distributed=False, shuffle=False)
print(f'Using dataset: {opt.dataconf} {opt.variant}', flush=True)

# output folder that stores pre-calculated correspondence vectors as PyTorch tensors
out_dir = 'data/megadepth/' + opt.variant + '_aliked/'
if not os.path.isdir(out_dir): os.makedirs(out_dir)

# setup detector
modelconf = {
	'name': 'two_view_pipeline',
	'extractor':
		{'name': 'extractors.aliked',
		'max_num_keypoints': opt.nfeatures,
		'detection_threshold': opt.conf_thresh},
	'matcher':
        {'name': 'matchers.lightglue_pretrained',
		'depth_confidence': -1,
		'width_confidence': -1,
        'features': 'aliked',
        'filter_threshold': 0.1,},
	'ground_truth':
		{'name': 'matchers.depth_matcher',
		'th_positive': 3,
		'th_negative': 5,
		'th_epi': 5},
	'allow_no_extract': True
}
modelconf = OmegaConf.create(modelconf)
model = get_model(modelconf.name)(modelconf).to('cuda:0')
model.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(dataloader)):
        scene = data['view0']['scene'][0]
        assert data['view0']['scene'][0] == data['view1']['scene'][0], "Scenes' names are not the same"
        img1_name = data['view0']['name'][0].split('.')[0]
        img2_name = data['view1']['name'][0].split('.')[0]

        tqdm.write("Processing pair %5d of %5d. (%30s, %30s)" % (i, len(dataloader), img1_name, img2_name))

        img1_shape = data['view0']['image_size'][0].numpy() # W, H
        img1_shape = np.array([img1_shape[1], img1_shape[0], 3]) # H, W, 3
        img2_shape = data['view1']['image_size'][0].numpy() # W, H
        img2_shape = np.array([img2_shape[1], img2_shape[0], 3]) # H, W, 3

        data['view0']['image'] = data['view0']['image'][:, :, :img1_shape[0], :img1_shape[1]]
        data['view1']['image'] = data['view1']['image'][:, :, :img2_shape[0], :img2_shape[1]]

        pred = model(batch_to_device(data, 'cuda:0', non_blocking=True))

        K1 = data['view0']['camera'].calibration_matrix()[0] # B * 3 * 3
        K2 = data['view1']['camera'].calibration_matrix()[0] # B * 3 * 3

        GT_R_Rel = data['T_0to1'].R[0]
        GT_t_Rel = data['T_0to1'].t[0].unsqueeze(-1)

        matches1 = pred['matches0'][0]
        mask1 = matches1 != -1
        pts1 = pred['keypoints0'][0][mask1] # 1 * N * 2
        pts2 = pred['keypoints1'][0][matches1[mask1]] # 1 * N * 2

        desc1 = pred['descriptors0'][0][mask1] # 1 * N * 256
        desc2 = pred['descriptors1'][0][matches1[mask1]] # 1 * N * 256

        idx = torch.cat([torch.arange(end=mask1.clone().detach().int().sum(), device=matches1.device).unsqueeze(-1),
                        torch.tensor(mask1[mask1.nonzero(as_tuple=True)], device=matches1.device).unsqueeze(-1)], dim=-1)
        ratios = torch.ones_like(pts1[..., 0]) # N

        # Patches start
        bias = torch.tensor([[opt.patch_radius]*2], device=pts1.device)
        pad_amount = opt.patch_radius
        
        if data['view0']['image'].shape[1] == 3:  # RGB
            scale = data['view0']['image'].new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            data['view0']['image'] = (data['view0']['image'] * scale).sum(1, keepdim=True)
            
            scale = data['view1']['image'].new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            data['view1']['image'] = (data['view1']['image'] * scale).sum(1, keepdim=True)

        assert (data['view0']['image'][0] < (1. + 1e-5)).all() and (data['view0']['image'][0] > -(1. + 1e-5)).all(), "Image 0 out of range"
        img1_padded = torch.nn.functional.pad(data['view0']['image'][0][:, :img1_shape[0], :img1_shape[1]], [pad_amount]*4, mode='constant', value=0.)
        patch1 = extract_patches(img1_padded, (pts1 - bias + pad_amount).to(device=img1_padded.device, dtype=torch.int32), 2*opt.patch_radius+1)[0]

        assert (data['view1']['image'][0] < (1. + 1e-5)).all() and (data['view1']['image'][0] > -(1. + 1e-5)).all(), "Image 1 out of range"
        img2_padded = torch.nn.functional.pad(data['view1']['image'][0][:, :img2_shape[0], :img2_shape[1]], [pad_amount]*4, mode='constant', value=0.)
        patch2 = extract_patches(img2_padded, (pts2 - bias + pad_amount).to(device=img2_padded.device, dtype=torch.int32), 2*opt.patch_radius+1)[0]

        score1_padded = torch.nn.functional.pad(pred['score_map0'][0][:, :img1_shape[0], :img1_shape[1]], [pad_amount]*4, mode='constant', value=0.)
        scorepatch1 = extract_patches(score1_padded, (pts1 - bias + pad_amount).int(), 2*opt.patch_radius+1)[0]
        score2_padded = torch.nn.functional.pad(pred['score_map1'][0][:, :img2_shape[0], :img2_shape[1]], [pad_amount]*4, mode='constant', value=0.)
        scorepatch2 = extract_patches(score2_padded, (pts2 - bias + pad_amount).int(), 2*opt.patch_radius+1)[0]
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
                scorepatch1.cpu().numpy().astype(np.float32),
                scorepatch2.cpu().numpy().astype(np.float32),
                desc1.cpu().numpy().astype(np.float32), # N, D
                desc2.cpu().numpy().astype(np.float32),  # N, D
                ], f)

# pts1:  (N, 2)
# pts2:  (N, 2)
# ratios:  (N)
# img1:  (H, W, 3)
# img2:  (H, W, 3)
# K1:  (3, 3)
# K2:  (3, 3)
# GT_R_Rel:  (3, 3)
# GT_t_Rel:  (3, 1)