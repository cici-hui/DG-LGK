import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import sys

from torch.autograd import Variable

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

pspnet_specs = {
    'n_classes': 19,
    'block_config': [3, 4, 23, 3],
}

'''
Basic blocks
'''

def load_url(url, model_dir='/home/wilson/RL/image_segmentation/code/v11/pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
