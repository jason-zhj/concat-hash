import os
import torch
import torch.nn as nn
from ml_toolkit.log_analyze import plot_trend_graph
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

def get_pairwise_sim_loss(feats,labels,normalize=True):
    "loss: sum(logp(hi,hj)), where p is defined by sigmoid function"
    total_log_prob = None
    for i in range(len(labels)):
        lbl_i = labels[i]
        feat_i = feats[i]

        for j in range(i+1,len(labels)):
            lbl_j = labels[j]
            feat_j = feats[j]

            if (lbl_i == lbl_j):
                # same label, try to maximize probability
                log_prob = torch.log(_get_sim_prob(feat_i,feat_j))
            else:
                # different label, try to minimize probability
                log_prob = torch.log(1-_get_sim_prob(feat_i,feat_j))

            total_log_prob = log_prob if total_log_prob is None else torch.add(total_log_prob, log_prob)

    num_pairs = len(labels) * (len(labels) - 1) / 2
    return torch.sum(-total_log_prob) / num_pairs if normalize else torch.sum(-total_log_prob)


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


def save_loss_records(loss_records,save_to, loss_name):
    lines = [str(l) for l in loss_records]
    plot_trend_graph(var_names=[loss_name], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title=loss_name, save_to=os.path.join(save_to, "{}.png".format(loss_name)), show_fig=False)