# model with dropout applied on both shared and specific

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def flatten(tensor):
    return tensor.view(tensor.data.size(0),-1)

def gen_hash_from_modules(models,params,imgs,use_specific_code,binary=True):
    "return the hash output"
    basic_feat_ext = models["basic_feat_ext"]
    shared_feat_gen = models["shared_feat_gen"]
    specific_feat_gen = models["specific_feat_gen"]
    images = Variable(imgs).float()
    if (torch.cuda.is_available()):
        images = images.cuda()
    basic_feats = basic_feat_ext(images)
    shared_feats = shared_feat_gen(basic_feats)
    specific_feats = specific_feat_gen(basic_feats)

    if (use_specific_code):
        return (torch.sign(shared_feats), torch.sign(specific_feats)) if binary else  (shared_feats, specific_feats)
    else:
        return torch.sign(shared_feats) if binary else shared_feats


def _calc_output_dim(models,input_dim):
    """
    :param models: a list of model objects
    :param input_dim: like [3,100,100]
    :return:  the output dimension (in one number) if an image of `input_dim` is passed into `models`
    """
    input_tensor = torch.from_numpy(np.zeros(input_dim))
    input_tensor.unsqueeze_(0)
    img = Variable(input_tensor).float()
    output = img
    for model in models:
        output = model(output)
    return output.data.view(output.data.size(0), -1).size(1)


class BasicFeatExtractor(nn.Module):
    def __init__(self,params):
        super(BasicFeatExtractor, self).__init__()

        self.basic_feat_extract = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )

    def forward(self,x):
        out = self.basic_feat_extract(x)
        return out

class SpecificFeatGen(nn.Module):
    "the size of the specific feature is the same as hash code"
    def __init__(self,params):
        super(SpecificFeatGen, self).__init__()
        self.use_dropout = params.use_dropout

        self.conv1 = nn.Sequential(
            # first convolution
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d()
        )

        self.conv2 = nn.Sequential(
            # second convolution
            nn.Conv2d(32, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )

        gen_output_dim = _calc_output_dim(models=[BasicFeatExtractor(params=params),self.conv1,self.conv2],
                                          input_dim=[3,params.image_scale, params.image_scale])

        # this size can be checked using specific_feat.data.size(1)
        self.l1 = nn.Linear(in_features=gen_output_dim, out_features=200)
        self.l1_bnm = nn.BatchNorm1d(200)
        self.l2 = nn.Linear(in_features=200, out_features=params.specific_hash_size)

    def forward(self,x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        specific_feat = flatten(conv2_out)
        l1_out = F.sigmoid(self.l1_bnm(self.l1(specific_feat)))
        return F.tanh(self.l2(l1_out))
