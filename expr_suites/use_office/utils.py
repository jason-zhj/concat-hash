import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torch.autograd import Variable

def save_models(models,save_model_to,save_obj=True,save_params=True):
    "models is a dict {name:model_obj}"
    for name,model in models.items():
        if (save_obj):
            torch.save(model, os.path.join(save_model_to, "{}.model".format(name)))
        if (save_params):
            torch.save(model.state_dict(), os.path.join(save_model_to, "{}.params".format(name)))


def get_data_loader(data_path,params):
    "return a torch data loader"
    # image loading
    preprocess = transforms.Compose([
        transforms.Scale(params.image_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])

    # create dataloader
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=preprocess)
    return  torch.utils.data.DataLoader(dataset,
                                                batch_size=params.batch_size, shuffle=params.shuffle_batch)


def _to_onehot(labels,num_classes=31):
    zeros = torch.zeros(len(labels), num_classes)
    return zeros.scatter_(1, labels.view(-1, 1), 1)


def _get_squared_dist(x, y):
    "(x-y)^2"
    return torch.pow(
        torch.add(x,-y),
        2
    )

def _get_rms_diff(x,y):
    p = torch.pow(
        torch.add(x,-y),
        2
    )
    vector_len = Variable(torch.FloatTensor([x.data.size(0) for _ in range(x.data.size(0))]), requires_grad=False)
    return torch.sqrt(torch.sum(p / vector_len))

def _get_sim_prob(x,y):
    "sigmoid(<x,y>)"
    return torch.sigmoid(
        torch.dot(x,y)
    )

def get_shared_feat_loss(src_feats,tgt_feats,src_labels,tgt_labels,use_sqrt=True,use_rms=True):
    "return a pairwise loss varaible, if margin is not None, max-margin will be applied on squared error"
    loss = None
    for i in range(len(src_labels)):
        src_lbl = src_labels[i]
        src_feat = src_feats[i]

        for j in range(len(tgt_labels)):
            tgt_lbl = tgt_labels[j]
            tgt_feat = tgt_feats[j]
            sq_dist = _get_squared_dist(src_feat, tgt_feat)
            if (use_sqrt):
                sq_dist = torch.sqrt(sq_dist)
            if (use_rms):
                sq_dist = _get_rms_diff(src_feat,tgt_feat)

            if (tgt_lbl == src_lbl):
                # same label, try to minimize feature distance
                error = sq_dist
                loss = error if loss is None else torch.add(loss,error)
            else:
                # different label, try to maximize feature distance
                error = -sq_dist
                loss = error if loss is None else torch.add(loss, error)

    return torch.div(torch.sum(loss),len(src_labels) * len(tgt_labels))


def get_pairwise_sim_loss(feats,labels,normalize=True):
    "loss: sum(logp(hi,hj)), where p is defined by sigmoid function"
    labels_onehot = _to_onehot(labels=labels)

    A_square = torch.mm(feats, feats.t())
    TINY = 10e-8
    A_square_sigmod = (torch.sigmoid(A_square) - 0.5) * (1 - TINY) + 0.5
    is_same_lbl = torch.mm(labels_onehot, labels_onehot.t())

    log_prob = torch.mul(torch.log(A_square_sigmod), is_same_lbl)
    log_prob += torch.mul(torch.log(1 - A_square_sigmod), 1 - is_same_lbl)
    sum_log_prob = (log_prob.sum() - log_prob.diag().sum()) / 2.0

    if (normalize):
        num_pairs = len(feats) * (len(feats) - 1) / 2
        return torch.sum(-sum_log_prob) / num_pairs
    else:
        return torch.sum(-sum_log_prob)


def get_crossdom_pairwise_sim_loss(src_feats,tgt_feats,src_labels,tgt_labels,normalize=True):
    "loss: sum(logp(hi,hj)), where p is defined by sigmoid function"
    total_log_prob = None
    for i in range(len(src_labels)):
        lbl_i = src_labels[i]
        feat_i = src_feats[i]

        for j in range(len(tgt_labels)):
            lbl_j = tgt_labels[j]
            feat_j = tgt_feats[j]

            if (lbl_i == lbl_j):
                # same label, try to maximize probability
                log_prob = torch.log(_get_sim_prob(feat_i, feat_j))
            else:
                # different label, try to minimize probability
                log_prob = torch.log(1 - _get_sim_prob(feat_i, feat_j))

            total_log_prob = log_prob if total_log_prob is None else torch.add(total_log_prob, log_prob)

    num_pairs = len(src_labels) * len(tgt_labels)
    return torch.sum(-total_log_prob) / num_pairs if normalize else torch.sum(-total_log_prob)

"""
a singleton static logger to be shared by all scripts
"""
import logging

class LoggerGenerator():
    logger_dict = {}

    @staticmethod
    def get_logger(log_file_path):
        if (log_file_path not in LoggerGenerator.logger_dict.keys() ):
            print("Creating a logger that writes to {}".format(log_file_path))
            logger = logging.getLogger('myapp-{}'.format(log_file_path))
            hdlr = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
            LoggerGenerator.logger_dict[log_file_path] = logger
            return logger
        else:
            # logger already created
            return LoggerGenerator.logger_dict[log_file_path]


def get_curtime_str():
    from datetime import datetime
    current_time = str(datetime.now())[:-7].replace(" ", "T").replace(":", "-")
    return current_time

def str_param_module(params):
    "return the string representation of the params module"
    filtered_items = {key: value for key, value in vars(params).items() if key.find("__") != 0}
    ls = []
    for key, value in filtered_items.items():
        ls.append("{}={}".format(key, value))

    return "\n".join(ls)

def save_param_module(params,save_to):
    with open(save_to,"w") as f:
        f.write(str_param_module(params=params))


def _get_quantization_loss(continuous_code):
    "continuous code is output of `torch.tanh`, which should range between [-1,1]"
    discrete_code = torch.sign(continuous_code)
    quantization_loss = torch.sum(
        torch.pow(
            torch.add(discrete_code, torch.neg(continuous_code)), 2
        )
    )
    return quantization_loss / len(continuous_code)


def _add_loss_term(ori_loss,add_term):
    if (ori_loss is None):
        return add_term
    else:
        return torch.add(ori_loss,add_term)