import os
import json
import torch
import numpy as np

def create_log_dir(result_path, opt):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    with open(result_path / 'config.json', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
        
    # path for saving traning logs
    opt.log_path = result_path / 'logs'
    opt.model_path = result_path / 'models'
    opt.res_path = result_path / 'pickled_results'

    if not os.path.isdir(opt.log_path):
        os.makedirs(opt.log_path)
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.isdir(opt.res_path):
        os.makedirs(opt.res_path)
    if os.path.exists(result_path / 'config.th'):
        print('warning: will overwrite config file')
    

def last_checkpoint(model_path):
    checkpoints = [f for f in os.listdir(model_path) if f.startswith('checkpoint') and f.endswith('.pth')]
    if len(checkpoints) == 0:
        return None
    checkpoints = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    last_checkpoint = max(checkpoints)
    return os.path.join(model_path, f'checkpoint_{last_checkpoint}.pth')

# Below function is borrowed from 
def normalize_pts(pts, im_size):
	"""Normalize image coordinate using the image size.

	Pre-processing of correspondences before passing them to the network to be 
	independent of image resolution.
	Re-scales points such that max image dimension goes from -0.5 to 0.5.
	In-place operation.

	Keyword arguments:
	pts -- 3-dim array conainting x and y coordinates in the last dimension, first dimension should have size 1.
	im_size -- image height and width
	"""	

	pts[:, 0] -= float(im_size[1]) / 2
	pts[:, 1] -= float(im_size[0]) / 2
	pts /= float(max(im_size[:2]))

# Below functions are borrowed from 
def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

# This function is borrowed from GlueFactory (https://github.com/cvg/glue-factory)
def extract_patches(
    tensor: torch.Tensor,
    required_corners: torch.Tensor,
    ps: int,
) -> torch.Tensor:
    c, h, w = tensor.shape
    corner = required_corners.long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1), corner.float()

# Below functions
def _homo(x):
    # input: x [N, 2] or [batch_size, N, 2]
    # output: x_homo [N, 3]  or [batch_size, N, 3]
    assert len(x.size()) in [2, 3]
    # print(f"x: {x.size()[0]}, {x.size()[1]}, {x.dtype}, {x.device}")
    if len(x.size())==2:
        ones = torch.ones(x.size()[0], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 1)
    elif len(x.size())==3:
        ones = torch.ones(x.size()[0], x.size()[1], 1, dtype=x.dtype, device=x.device)
        x_homo = torch.cat((x, ones), 2)
    return x_homo

def _sampson_dist(F, X, Y, if_homo=False):
    if not if_homo:
        X = _homo(X)
        Y = _homo(Y)
    if len(X.size())==2:
        nominator = (torch.diag(Y@F@X.t()))**2
        Fx1 = torch.mm(F, X.t())
        Fx2 = torch.mm(F.t(), Y.t())
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    else:
        nominator = (torch.diagonal(Y@F@X.transpose(1, 2), dim1=1, dim2=2))**2
        Fx1 = torch.matmul(F, X.transpose(1, 2))
        Fx2 = torch.matmul(F.transpose(1, 2), Y.transpose(1, 2))
        denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Fx2[:, 0]**2 + Fx2[:, 1]**2

    errors = nominator/denom
    return errors

def calculate_loss(inp, gt_E, K1s, K2s, train_thr):
    loss = _sampson_dist(gt_E, inp[...,:2], inp[...,2:4])
    threshold = train_thr/((K1s[:, 0, 0] + K1s[:, 1, 1] +K2s[:, 0, 0] + K2s[:, 1, 1])/4)
    threshold = threshold.view(-1, 1)
    thresholdsq = threshold ** 2.

    # squared msac score with essential error
    loss = torch.where(loss < thresholdsq, loss/thresholdsq, torch.ones_like(loss))
    loss = loss.mean()

    return loss