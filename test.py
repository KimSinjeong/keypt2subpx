import numpy as np
import cv2

import torch
import pickle
from tqdm import tqdm
from pathlib import Path

import pygcransac

from model import AttnTuner
from utils import compute_pose_error, pose_auc, calculate_loss, create_log_dir
from logger import Logger
from dataset import SparsePatchDataset
from settings import get_config, get_logger, print_usage

CUDA_LAUNCH_BLOCKING=2, 3
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def valid(valid_loader, model, opt, writer=None, step=-1):
    model.eval()
    err_ts, err_Rs = [], []
    loss_vals = []
    inlier_cnt = []
    outlier_cnt = []

    with torch.no_grad():
        for idx, valid_data in enumerate(tqdm(valid_loader)):
            corr, gt_F, gt_E, gt_R, gt_t, K1s, K2s, im_size1, im_size2, patch1, patch2, descriptor1, descriptor2 = valid_data[:13]
            scorepatch1, scorepatch2 = (valid_data[13].cuda(), valid_data[14].cuda()) if len(valid_data) > 13 else (None, None)

            ret = None
            if len(corr.shape) > 1 and corr.shape[1] > 4:

                if opt.vanilla: # without our model
                    inp = corr[...,:4].cuda() # B x N x 4
                else: # with our model
                    meanft = ((descriptor1 + descriptor2) / 2.).cuda()
                    idx1 = 2.5* K1s[:,:2,:2].unsqueeze(1).cuda().inverse() @ model(patch1.cuda(), scorepatch1, meanft).unsqueeze(-1)
                    idx1 = idx1.squeeze(-1)
                    idx2 = 2.5* K2s[:,:2,:2].unsqueeze(1).cuda().inverse() @ model(patch2.cuda(), scorepatch2, meanft).unsqueeze(-1)
                    idx2 = idx2.squeeze(-1)
                    inp = corr[...,:4].cuda() + torch.cat([idx1, idx2], 2) # B x N x 4

                K1s_cuda = K1s.cuda()
                K2s_cuda = K2s.cuda()

                kpts0 = torch.cat([inp.squeeze()[:, :2], torch.ones_like(inp.squeeze()[:, 0:1])], dim=-1)
                kpts1 = torch.cat([inp.squeeze()[:, 2:], torch.ones_like(inp.squeeze()[:, 0:1])], dim=-1)
                # Let's change mkpts' normalized points to image coordinates

                mkpts0 = kpts0@K1s_cuda.transpose(-1, -2).squeeze(0)
                mkpts1 = kpts1@K2s_cuda.transpose(-1, -2).squeeze(0)

                mkpts0 = mkpts0[:, :2] / mkpts0[:, 2:]
                mkpts1 = mkpts1[:, :2] / mkpts1[:, 2:]

                mkpts = torch.cat([mkpts0, mkpts1], dim=-1).cpu().detach().numpy()

                # Let's change mkpts' normalized points to image coordinates
                im_size1 = im_size1.detach().numpy()[0]
                im_size2 = im_size2.detach().numpy()[0]
                K1 = K1s.detach().numpy()[0]
                K2 = K2s.detach().numpy()[0]
                E, mask = pygcransac.findEssentialMatrix(
                    np.ascontiguousarray(mkpts), 
                    K1, K2,
                    im_size1[0], im_size1[1], im_size2[0], im_size2[1],
                    probabilities = [],
                    threshold = opt.ransac_thr,
                    conf = 0.99999, # RANSAC confidence
                    max_iters = 1000,
                    min_iters = 1000,
                    sampler = 0)

                kpts0 = kpts0[:, :2].cpu().detach().numpy()
                kpts1 = kpts1[:, :2].cpu().detach().numpy()

                mask = np.expand_dims(mask, axis=-1).astype(np.uint8)

                best_num_inliers = 0
                if E is not None:
                    for _E in np.split(E, len(E) / 3):
                        n, R, t, mask_ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
                        if n > best_num_inliers:
                            best_num_inliers = n
                            ret = (R, t[:, 0], mask.ravel() > 0)

                loss = calculate_loss(inp, gt_E.cuda(), K1s_cuda, K2s_cuda, opt.train_thr)
                tmp = loss.cpu().detach().item()
                if np.isnan(tmp):
                    tqdm.write(f"Pair #{idx}: Loss is nan!!")
                else:
                    loss_vals.append(tmp)
                
            if ret is None:
                tqdm.write(f"Pair #{idx}: Not enough points for E estimation or just E estimation failed")
                err_t, err_R = np.inf, np.inf
                inlier_cnt.append(0)
                outlier_cnt.append(corr.shape[1])
            else:
                R, t, inliers = ret
                T_0to1 = torch.cat([gt_R.squeeze(), gt_t.squeeze().unsqueeze(-1)], dim=-1).numpy()
                err_t, err_R = compute_pose_error(T_0to1, R, t)
                n = np.sum(inliers)
                outlier_cnt.append(len(mkpts) - n)
                inlier_cnt.append(n)

            err_ts.append(err_t)
            err_Rs.append(err_R)

    # Write the evaluation results to disk.
    out_eval = {
        'error_t': err_ts,
        'error_R': err_Rs,
        'inliers': inlier_cnt,
        'outliers': outlier_cnt
    }

    if opt.test:
        pickle.dump(out_eval, open(opt.res_path / f'results_{opt.current_split}_{opt.total_split}.pkl', 'wb'))

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        if pose_error == np.inf:
            pose_error = 180
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    in_rate = 100. * np.sum(inlier_cnt) / (np.sum(inlier_cnt) + np.sum(outlier_cnt))
    err_mean = np.mean(pose_errors)
    err_median = np.median(pose_errors)
    loss = np.mean(loss_vals)

    if writer is not None:
        writer.add_scalar('val/loss', loss, step)
        writer.add_scalar('val/inlier_ratio', np.sum(inlier_cnt) / (np.sum(inlier_cnt) + np.sum(outlier_cnt)), step)
        writer.add_scalar('val/AUC@5', aucs[0], step)
        writer.add_scalar('val/AUC@10', aucs[1], step)
        writer.add_scalar('val/AUC@20', aucs[2], step)

    print(f'Evaluation Results (mean over {len(err_ts)} pairs):')
    print('AUC@5\t AUC@10\t AUC@20\t InRat.\t Mean\t Median\t Avg loss\t')
    print(f'{aucs[0]:.2f}\t {aucs[1]:.2f}\t {aucs[2]:.2f}\t '
          f'{in_rate:.2f}\t {err_mean:.2f}\t {err_median:.2f}\t {loss:.8f}')

    return aucs[0], aucs[1], aucs[2], in_rate, err_mean, err_median, loss

if __name__ == "__main__":
    # parse command line arguments
    # If we have unparsed arguments, print usage and exit
    opt, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    logger = get_logger(opt)

    # construct folder that should contain pre-calculated correspondences
    test_data = opt.datasets.split(',') #support multiple training datasets used jointly
    logger.info('Using datasets: [' + ', '.join(test_data) + ']')

    # create or load model
    lentable = {
        'spnn': 256,
        'splg': 256,
        'aliked': 128,
        'dedode': 256,
        'xfeat': 64
    }
    with_score = opt.detector in ['spnn', 'splg', 'aliked']
    model = AttnTuner(lentable[opt.detector], with_score)
    if len(opt.model) > 0:
        model.load_state_dict(torch.load(opt.model)['model'])
    model = model.cuda()
    model.eval()

    # ----------------------------------------
    logger.info(f"Starting experiment {opt.experiment}")
    output_dir = Path(__file__).parent / 'results' / opt.experiment
    output_dir.mkdir(exist_ok=True, parents=True)
    create_log_dir(output_dir, opt)

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if opt.current_split > 1 and opt.total_split > 1:
        print(f"Working with split: {opt.current_split} / {opt.total_split}", flush=True)

    filelogger = Logger(opt.log_path / 'log_evaluation.txt', title='k2s')
    filelogger.set_names(['AUC5'] + ['AUC10', 'AUC20', 'InRat.', 'Mean', 'Median', 'Loss'])
    
    if opt.test: # Metrics: aucs5, aucs10, aucs20, in_rat, err_mean, err_median, loss_avg
        test_data = ['data/' + ds + '/' + 'test_' + opt.detector + '/' for ds in test_data]
        testset = SparsePatchDataset(test_data, opt.nfeatures, opt.fmat, not opt.sideinfo, not with_score,#False,
                current_split=opt.current_split, total_split=opt.total_split)
        test_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6, batch_size=1)
        aucs5, aucs10, aucs20, in_rat, err_mean, err_median, loss_avg = valid(test_loader, model, opt)
        filelogger.append([aucs5, aucs10, aucs20, in_rat, err_mean, err_median, loss_avg])
    else:
        val_data = ['data/' + ds + '/' + 'val_' + opt.detector + '/' for ds in test_data]
        valset = SparsePatchDataset(val_data, opt.nfeatures, opt.fmat, not opt.sideinfo, not with_score,#False,
                current_split=opt.current_split, total_split=opt.total_split)
        val_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=6, batch_size=1)
        all_metrics = None
        for i in range(opt.total_run):
            logger.info(f"Test run {i+1}/{opt.total_run}")
            metrics = valid(val_loader, model, opt)
            filelogger.append(metrics)
            if all_metrics is None:
                all_metrics = [[m] for m in metrics]
            else:
                for j in range(len(all_metrics)):
                    all_metrics[j].append(metrics[j])
        all_metrics = [np.mean(m) for m in all_metrics]

        print('----------------------------------------')
        print(f'Final Evaluation Results (average over {opt.total_run} runs):')
        print('AUC@5\t AUC@10\t AUC@20\t InRat.\t Mean\t Median\t Avg loss\t')
        print(f'{all_metrics[0]:.2f}\t {all_metrics[1]:.2f}\t {all_metrics[2]:.2f}\t '
            f'{all_metrics[3]:.2f}\t {all_metrics[4]:.2f}\t {all_metrics[5]:.2f}\t {all_metrics[6]:.8f}')