import numpy as np
import torch
import os
import cv2
import math
import utils
import pickle

from torch.utils.data import Dataset

class SparsePatchDataset(Dataset):
	"""Sparse correspondences dataset."""

	def __init__(self, folders, nfeatures, fmat=False, overwrite_side_info=False, without_score=False, without_depth=True, train=False, total_split=-1, current_split=-1):
		self.nfeatures = nfeatures # ensure fixed number of features, -1 keeps original feature count
		self.overwrite_side_info = overwrite_side_info # if true, provide no side information to the neural guidance network
		self.without_score = without_score
		self.without_depth = without_depth
		self.train = train
		
		# collect precalculated correspondences of all provided datasets
		self.files = []
		for folder in folders:
			self.files += [folder + f for f in os.listdir(folder)]
		
		if total_split > 0 and current_split > 0:
			start_iter = len(self.files) * (current_split-1) // total_split
			end_iter = len(self.files) * current_split // total_split
			self.files = self.files[start_iter:end_iter]

		self.fmat = fmat # estimate fundamental matrix instead of essential matrix
		self.minset = 5 # minimal set size for essential matrices
		if fmat: self.minset = 7 # minimal set size for fundamental matrices
			
	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):

		# load precalculated correspondences using pickle
		data = None
		with open(self.files[idx], 'rb') as f:
			data = pickle.load(f)
		assert data is not None, f"Error: Correspondences {self.files[idx]} could not be loaded."

		# correspondence coordinates and matching ratios (side information)
		pts1, pts2, ratios = data[0], data[1], data[2] # N x 2, N x 2, N
		# image sizes
		im_size1, im_size2 = torch.from_numpy(np.asarray(data[3])), torch.from_numpy(np.asarray(data[4]))
		# image calibration parameters
		K1, K2 = torch.from_numpy(data[5]), torch.from_numpy(data[6])
		# ground truth pose
		gt_R, gt_t = torch.from_numpy(data[7]), torch.from_numpy(data[8])

		patch1, patch2 = torch.from_numpy(data[9]), torch.from_numpy(data[10])
		if not self.without_score:
			scorepatch1, scorepatch2 = torch.from_numpy(data[11]), torch.from_numpy(data[12])
			descriptor1, descriptor2 = torch.from_numpy(data[13]), torch.from_numpy(data[14])
		else:
			descriptor1, descriptor2 = torch.from_numpy(data[11]), torch.from_numpy(data[12])
		
		if not self.without_depth:
			depth1, depth2 = torch.from_numpy(data[-2]), torch.from_numpy(data[-1])
		
		if pts1.shape[0] < self.minset:
			
			if self.train:
				print("WARNING! Not enough correspondences left. Only %d correspondences among would be left, so I instead sample another one." % (int(pts1.shape[1])))
				return self.__getitem__(np.random.randint(0, len(self.files)))

			else:
				print("WARNING! Not enough correspondences left. Only %d correspondences among would be left, so I instead return Nones." % (int(pts1.shape[1])))
				nums = 13
				if not self.without_score:
					nums += 2
				if not self.without_depth:
					nums += 2
				return [np.zeros(0)]*nums
		
		if self.overwrite_side_info:
			ratios = np.zeros(ratios.shape, dtype=np.float32)

		if self.fmat:
			# for fundamental matrices, normalize image coordinates using the image size (network should be independent to resolution)
			utils.normalize_pts(pts1, im_size1)
			utils.normalize_pts(pts2, im_size2)
		else:
			#for essential matrices, normalize image coordinate using the calibration parameters
			pts1 = cv2.undistortPoints(pts1[None], K1.numpy(), None).squeeze(1)
			pts2 = cv2.undistortPoints(pts2[None], K2.numpy(), None).squeeze(1)

		# stack image coordinates and side information into one tensor
		correspondences = np.concatenate((pts1, pts2, ratios[...,None]), axis=-1)
		# correspondences = np.transpose(correspondences)
		correspondences = torch.from_numpy(correspondences)

		indices = None
		if self.nfeatures > 0:
			# ensure that there are exactly nfeatures entries in the data tensor 
			if correspondences.size(0) > self.nfeatures:
				rnd = torch.randperm(correspondences.size(0))
				correspondences = correspondences[rnd]
				correspondences = correspondences[0:self.nfeatures]
				indices = rnd[0:self.nfeatures]
			else:
				indices = torch.arange(0, correspondences.size(0))

			if correspondences.size(0) < self.nfeatures:
				result = correspondences
				for i in range(0, math.ceil(self.nfeatures / correspondences.size(0) - 1)):
					rnd = torch.randperm(correspondences.size(0))
					result = torch.cat((result, correspondences[rnd]), dim=0)
					indices = torch.cat((indices, rnd))
				correspondences = result[0:self.nfeatures]
				indices = indices[0:self.nfeatures]

		# construct the ground truth essential matrix from the ground truth relative pose
		gt_E = torch.zeros((3,3))
		gt_E[0, 1] = -float(gt_t[2,0])
		gt_E[0, 2] = float(gt_t[1,0])
		gt_E[1, 0] = float(gt_t[2,0])
		gt_E[1, 2] = -float(gt_t[0,0])
		gt_E[2, 0] = -float(gt_t[1,0])
		gt_E[2, 1] = float(gt_t[0,0])

		gt_E = gt_E.mm(gt_R)

		# fundamental matrix from essential matrix
		gt_F = K2.inverse().transpose(0, 1).mm(gt_E).mm(K1.inverse())

		if indices is not None:
			patch1, patch2 = patch1[indices], patch2[indices]
			if not self.without_score:
				scorepatch1, scorepatch2 = scorepatch1[indices], scorepatch2[indices]
			descriptor1, descriptor2 = descriptor1[indices], descriptor2[indices]
			if not self.without_depth:
				depth1, depth2 = depth1[indices], depth2[indices]

		retlist = [correspondences, gt_F, gt_E, gt_R, gt_t, K1, K2, im_size1, im_size2, patch1, patch2, descriptor1, descriptor2]
		if not self.without_score:
			retlist += [scorepatch1, scorepatch2]
		if not self.without_depth:
			retlist += [depth1, depth2]

		return retlist
