"""
Utility functions for use in testing
"""
import numpy as np
import torch
import torchvision
import os

from torch.autograd import Variable
from torchvision import transforms


def _save_hash_code(item_set,fns,save_to,delimiter=","):
    "item_set should be list of {label:...,hash:...}"
    write_ls = []
    # write in the form of "filename\tlabel\thash"
    for i, fn in enumerate(fns):
        write_ls.append("{}{}{}{}{}".format(fn,delimiter,item_set[i]["label"],delimiter,item_set[i]["hash"]))
    open(save_to,"w").write("\n".join(write_ls))


def _get_data_loader(path, params, shuffle=False, use_batch=True):
    # image loading
    preprocess = transforms.Compose([
        transforms.Scale(params.image_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])

    dataset = torchvision.datasets.ImageFolder(root=path, transform=preprocess)
    if (use_batch):
        batch_size = params.test_batch_size
    else:
        batch_size = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    img_filenames = [t[0] for t in dataset.imgs]
    return loader, img_filenames


def _construct_hash_function(models, params, use_specific_code, use_shared_code,model_def,binary=True):
    "return a function that takes in images and return hash code, if `binary` is False, will return real-value outputs delimited by comma"
    def hash_function(imgs):
        ## only use specific code
        basic_feat_ext = models["basic_feat_ext"]
        specific_feat_gen = models["specific_feat_gen"]
        images = Variable(imgs).float()
        if (torch.cuda.is_available()):
            images = images.cuda()
        basic_feats = basic_feat_ext(images)
        specific_feats = specific_feat_gen(basic_feats)
        hash_outputs = torch.sign(specific_feats)

        hash_str_ls = []
        for output in hash_outputs:
            output = output.data.cpu().numpy().astype(np.int8)
            output[output == -1] = 0
            hash_str = "".join(output.astype(np.str))
            hash_str_ls.append(hash_str)
        return hash_str_ls

    return hash_function


def _create_label_hash_dicts(hash_ls, label_ls):
    "return a list of dict {'label':...,'hash':'001010...'}"
    assert len(hash_ls) == len(label_ls)
    return [
        {"label":label_ls[i],"hash":hash_ls[i]}
        for i in range(len(hash_ls))
    ]


def _model_files_exist(path):
    files = ["basic_feat_ext.model","specific_feat_gen.model"]
    return all([os.path.exists(os.path.join(path,f)) for f in files])

def _load_models_from_path(saved_model_path, params, model_def_module, test_mode = True,use_model_file=True):
    "return a dict {'basic_feat_ext':model_obj, 'shared_feat_gen': .., 'specific_feat_gen':..}"
    if (_model_files_exist(path=saved_model_path) and use_model_file):
        print("Loading models using .model files")
        basic_ext = torch.load("{}/{}".format(saved_model_path,"basic_feat_ext.model"))
        specific_gen = torch.load("{}/{}".format(saved_model_path,"specific_feat_gen.model"))
    else:
        print("Loading models using .params files")
        basic_ext = model_def_module.BasicFeatExtractor(params=params)
        basic_ext.load_state_dict(
            torch.load("{}/{}".format(saved_model_path, "basic_feat_ext.params"), map_location={'cuda:0': 'cpu'}))
        specific_gen = model_def_module.SpecificFeatGen(params=params)
        specific_gen.load_state_dict(
            torch.load("{}/{}".format(saved_model_path, "specific_feat_gen.params"), map_location={'cuda:0': 'cpu'}))
    # set models to eval mode if we are doing testing
    if (test_mode):
        basic_ext.eval()
        specific_gen.eval()
        print("load models in eval mode")

    models = {
        "basic_feat_ext": basic_ext,
        "specific_feat_gen": specific_gen
    }
    return models