import argparse
import pickle
import numpy as np
import glob

from utils import pose_auc

parser = argparse.ArgumentParser(
    description="Summarize Test Set Results.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('experiment', type=str, help='experiment name')

opt, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    parser.print_usage()
    exit(1)

filelist = glob.glob(f"results/{opt.experiment}/pickled_results/*.pkl")

err_t, err_r, inliers, outliers = [], [], [], []
for filename in filelist:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        err_t += data['error_t']
        err_r += data['error_R']
        inliers += data['inliers']
        outliers += data['outliers']

pose_errors = []
for idx in range(len(err_t)):
    pose_error = np.maximum(err_t[idx], err_r[idx])
    if pose_error == np.inf:
        pose_error = 180
    pose_errors.append(pose_error)

thresholds = [5, 10, 20]
aucs = pose_auc(pose_errors, thresholds)
aucs = [100.*yy for yy in aucs]

print('Evaluation Results (mean over {} pairs):'.format(len(err_t)))
print('AUC@5\t AUC@10\t AUC@20\tInRat.\tMean\tMedian\t')
print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}'.format(
    aucs[0], aucs[1], aucs[2], 100.*np.sum(inliers) / (np.sum(inliers) + np.sum(outliers)),
    np.mean(pose_errors), np.median(pose_errors)))
