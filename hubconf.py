dependencies = ['torch']
from model import Keypt2Subpx as Keypt2Subpx_
import torch

lentable = {
    'spnn': 256,
    'splg': 256,
    'aliked': 128,
    'dedode': 256,
    'xfeat': 64
}

scoreusetable = {
    'spnn': True,
    'splg': True,
    'aliked': True,
    'dedode': False,
    'xfeat': False
}

def Keypt2Subpx(pretrained=True, detector='splg'):
    """
        Keypt2Subpx model
        pretrained (bool): kwargs, load pretrained weights into the model
        detector (str): kwargs, detector-matcher combination; supported options: 'spnn', 'splg', 'aliked', 'dedode', 'xfeat'
    """
    assert detector in lentable.keys(), f"Unsupported detector-matcher combination; Supported options: {' '.join(lentable.keys())}"

    model = Keypt2Subpx_(lentable[detector], scoreusetable[detector])

    weights = None
    if pretrained:
        weights = torch.hub.load_state_dict_from_url(
            f"https://github.com/KimSinjeong/keypt2subpx/raw/master/pretrained/k2s_{detector}_pretrained.pth",
            map_location=torch.device('cpu'))
        model.net.load_state_dict(weights['model'])
    
    return model