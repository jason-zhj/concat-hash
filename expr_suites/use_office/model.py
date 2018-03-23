# model for office

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchvision import models

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

    num_vgg_layers_used = 18

    def __init__(self):
        super(BasicFeatExtractor, self).__init__()
        vgg_features = models.vgg11_bn(pretrained=True).features
        self.basic_feat_extract = torch.nn.Sequential()
        for x in range(self.num_vgg_layers_used):
            self.basic_feat_extract.add_module(str(x), vgg_features[x])

    def forward(self,x):
        out = self.basic_feat_extract(x)
        return out

class SharedFeatGen(nn.Module):
    "the size of the shared feature is the same as hash code"
    def __init__(self,params):
        super(SharedFeatGen, self).__init__()

        self.conv1 = nn.Sequential(
            # first convolution
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Dropout2d()
        )

        self.conv2 = nn.Sequential(
            # second convolution
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )

        gen_output_dim = _calc_output_dim(models=[BasicFeatExtractor(),self.conv1,self.conv2],
                                          input_dim=[3,params.image_scale,params.image_scale])

        # this size can be checked using shared_feat.data.size(1)
        self.l1 = nn.Linear(in_features=gen_output_dim, out_features=200)
        self.l1_bnm = nn.BatchNorm1d(200)
        self.l2 = nn.Linear(in_features=200, out_features=params.hash_size)

    def forward(self,x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        specific_feat = flatten(conv2_out)
        l1_out = F.sigmoid(self.l1_bnm(self.l1(specific_feat)))
        return F.tanh(self.l2(l1_out))

class SpecificFeatGen(nn.Module):
    "the size of the specific feature is the same as hash code"
    def __init__(self,params):
        super(SpecificFeatGen, self).__init__()
        self.use_dropout = params.use_dropout

        self.conv1 = nn.Sequential(
            # first convolution
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Dropout2d()
        )

        self.conv2 = nn.Sequential(
            # second convolution
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=2, stride=2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d()
        )

        gen_output_dim = _calc_output_dim(models=[BasicFeatExtractor(),self.conv1,self.conv2],
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
