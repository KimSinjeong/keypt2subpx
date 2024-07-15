import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils import extract_patches

def create_meshgrid(
        x: torch.Tensor,
        normalized_coordinates: Optional[bool]) -> torch.Tensor:
    assert len(x.shape) == 4, x.shape
    _, _, height, width = x.shape
    _device, _dtype = x.device, x.dtype
    if normalized_coordinates:
        xs = torch.linspace(-1.0, 1.0, width, device=_device, dtype=_dtype)
        ys = torch.linspace(-1.0, 1.0, height, device=_device, dtype=_dtype)
    else:
        xs = torch.linspace(0, width - 1, width, device=_device, dtype=_dtype)
        ys = torch.linspace(0, height - 1, height, device=_device, dtype=_dtype)
    return torch.meshgrid(ys, xs)  # pos_y, pos_x

class SpatialSoftArgmax2d(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum(
            (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x: torch.Tensor = torch.sum(
            (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2

class AttnTuner(nn.Module):
	def __init__(self, output_dim=256, use_score=True):
		super(AttnTuner, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.feat_axis = 1
		self.normalized_coordinates = False
		self.use_score = use_score

		c1, c2, c3 = 16, 64, output_dim
		self.conv1a = nn.Conv2d(2 if use_score else 1, c1, kernel_size=3, stride=1, padding=0)
		self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
		self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=0)
		self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=0)
		self.logsoftargmax = SpatialSoftArgmax2d(self.normalized_coordinates)

	def forward(self, patch, scorepatch, desc):
		B, N, C, H, W = patch.shape
		B, N, F_ = desc.shape
		assert H == W, "Patch shape must be square"
		# P = (H // 2 +1) // 2 +1
		P = H-6

		patch = patch.view(B*N, C, H, W)
		if self.use_score:
			scorepatch = scorepatch.view(B*N, 1, H, W)
		desc = desc.view(B*N, F_, 1, 1)

		if patch.shape[1] == 3:  # RGB
			scale = patch.new_tensor([0.299, 0.587, 0.114]).view(*([1]*self.feat_axis), 3, 1, 1)
			patch = (patch * scale).sum(self.feat_axis, keepdim=True)
		x = torch.cat([patch, scorepatch], self.feat_axis) if self.use_score else patch

		# Shared Encoder
		x = self.relu(self.conv1a(x))
		x = self.relu(self.conv1b(x))
		x = self.relu(self.conv2a(x))
		x = self.relu(self.conv2b(x))
		x = self.conv3(x)
		x = F.normalize(x, p=2, dim=self.feat_axis)
		x = (x * desc).sum(dim=self.feat_axis).view(B, N, P, P) # Cosine similarity (in [-1, 1])    

		coord = self.logsoftargmax(x) - (P-1)/2.
		return coord # B x N x 2

class Keypt2Subpx(nn.Module):
    def __init__(self, output_dim=256, use_score=True):
        super(Keypt2Subpx, self).__init__()
        self.net = AttnTuner(output_dim, use_score)
        self.use_score = use_score
        self.patch_radius = 5

    def forward(self, keypt1, keypt2, img1, img2, desc1, desc2, score1=None, score2=None):
        """
            keypt1, keypt2: N x 2
            img1, img2: C x H x W
            score1, score2: 1 x H x W
            desc1, desc2: N x D
            TODO: Batch support
        """
        assert (img1 < (1. + 1e-5)).all() and (img1 > -(1. + 1e-5)).all(), "Image 1 out of range"
        assert (img2 < (1. + 1e-5)).all() and (img2 > -(1. + 1e-5)).all(), "Image 2 out of range"
        assert desc1 is not None and desc2 is not None
        C, H, W = img1.shape
        N, D = desc1.shape
        assert N == keypt1.shape[0] == keypt2.shape[0], "Number of keypoints mismatch with descriptors"

        bias = torch.tensor([[self.patch_radius]*2], device=keypt1.device)

        # RGB image to grayscale
        if img1.shape[0] == 3: # RGB
            scale = img1.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            img1 = (img1 * scale).sum(0, keepdim=True)
            
            scale = img2.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            img2 = (img2 * scale).sum(0, keepdim=True)

        # Image patches
        img1_padded = torch.nn.functional.pad(img1, [self.patch_radius]*4, mode='constant', value=0.)
        idx1 = (keypt1 - bias + self.patch_radius).int()
        patch1 = extract_patches(img1_padded.to(device=idx1.device), idx1, 2*self.patch_radius+1)[0].unsqueeze(0)

        img2_padded = torch.nn.functional.pad(img2, [self.patch_radius]*4, mode='constant', value=0.)
        idx2 = (keypt2 - bias + self.patch_radius).int()
        patch2 = extract_patches(img2_padded.to(device=idx2.device), idx2, 2*self.patch_radius+1)[0].unsqueeze(0)

        # Score patches
        scorepatch1, scorepatch2 = None, None
        if self.use_score:
            score1_padded = torch.nn.functional.pad(score1, [self.patch_radius]*4, mode='constant', value=0.)
            scorepatch1 = extract_patches(score1_padded.to(device=idx1.device), idx1, 2*self.patch_radius+1)[0].unsqueeze(0)

            score2_padded = torch.nn.functional.pad(score2, [self.patch_radius]*4, mode='constant', value=0.)
            scorepatch2 = extract_patches(score2_padded.to(device=idx2.device), idx2, 2*self.patch_radius+1)[0].unsqueeze(0)
        
        meanft = ((desc1 + desc2) / 2.).unsqueeze(0)

        coord1 = 2.5* self.net(patch1, scorepatch1, meanft).unsqueeze(-1) # 1 x N x 2 x 1
        # coord1 = 2.5* K1[:2,:2].view(1, 1, 2, 2).cuda().inverse() @ coord1 # 1 x N x 2 x 1
        coord1 = coord1.view(N, 2)

        coord2 = 2.5* self.net(patch2, scorepatch2, meanft).unsqueeze(-1) # 1 x N x 2 x 1
        # coord2 = 2.5* K2[:2,:2].view(1, 1, 2, 2).cuda().inverse() @ coord2 # 1 x N x 2 x 1
        coord2 = coord2.view(N, 2)

        return keypt1 + coord1, keypt2 + coord2