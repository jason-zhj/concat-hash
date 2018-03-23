"""
Utility functions for use in testing
"""
import numpy as np
import torch
import torchvision
import os
from ml_toolkit.data_process import batch_generator
from ml_toolkit.log_analyze import plot_trend_graph
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

def save_test_results(test_results,save_to):
    # create folder
    if (not os.path.exists(save_to)): os.makedirs(save_to)
    # save
    test_results["precision-recall-curve"].save(os.path.join(save_to, "precision-recall.png"))
    with open(os.path.join(save_to, "metrics.txt"), "w") as f:
        f.write(str(test_results["precision-recall-results"]))


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
        if (not use_specific_code):
            ## only use shared code
            hash_outputs = model_def.gen_hash_from_modules(models=models,imgs=imgs,params=params,use_specific_code=use_specific_code,binary=binary)
            hash_str_ls = []
            for output in hash_outputs:
                if (binary):
                    output = output.data.cpu().numpy().astype(np.int8)
                    output[output == -1] = 0
                    hash_str = "".join(output.astype(np.str))
                else:
                    hash_str = ",".join(output.astype(np.str))
                hash_str_ls.append(hash_str)
            return hash_str_ls
        else:
            ## concatenate shared and specific code
            shared_outputs, specific_outputs = model_def.gen_hash_from_modules(models=models,imgs=imgs,params=params,use_specific_code=use_specific_code,binary=binary)
            hash_str_ls = []
            for i in range(len(shared_outputs)):
                shared_out = shared_outputs[i].data.cpu().numpy()
                specific_out = specific_outputs[i].data.cpu().numpy()
                if (binary):
                    shared_out = shared_out.astype(np.int8)
                    specific_out = specific_out.astype(np.int8)
                    shared_out[shared_out == -1] = 0
                    specific_out[specific_out == -1] = 0
                    if (use_shared_code):
                        hash_str = "".join(shared_out.astype(np.str)) + "".join(specific_out.astype(np.str))
                    else: #NOTE: if shared code is not used, specific code has to be used
                        hash_str = "".join(specific_out.astype(np.str))
                else:
                    shared_str = ",".join(shared_out.astype(np.str))
                    specific_str = ",".join(specific_out.astype(np.str))
                    hash_str = shared_str + "," + specific_str if use_shared_code else specific_str
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
    files = ["basic_feat_ext.model","shared_feat_gen.model","specific_feat_gen.model"]
    return all([os.path.exists(os.path.join(path,f)) for f in files])

def _load_models_from_path(saved_model_path, params, model_def_module, test_mode = True,use_model_file=True):
    "return a dict {'basic_feat_ext':model_obj, 'shared_feat_gen': .., 'specific_feat_gen':..}"
    if (_model_files_exist(path=saved_model_path) and use_model_file):
        print("Loading models using .model files")
        basic_ext = torch.load("{}/{}".format(saved_model_path,"basic_feat_ext.model"))
        shared_gen = torch.load("{}/{}".format(saved_model_path, "shared_feat_gen.model"))
        specific_gen = torch.load("{}/{}".format(saved_model_path,"specific_feat_gen.model"))
    else:
        print("Loading models using .params files")
        basic_ext = model_def_module.BasicFeatExtractor(params=params)
        basic_ext.load_state_dict(
            torch.load("{}/{}".format(saved_model_path, "basic_feat_ext.params"), map_location={'cuda:0': 'cpu'}))
        shared_gen = model_def_module.SharedFeatGen(params=params)
        shared_gen.load_state_dict(
            torch.load("{}/{}".format(saved_model_path, "shared_feat_gen.params"), map_location={'cuda:0': 'cpu'}))
        specific_gen = model_def_module.SpecificFeatGen(params=params)
        specific_gen.load_state_dict(
            torch.load("{}/{}".format(saved_model_path, "specific_feat_gen.params"), map_location={'cuda:0': 'cpu'}))
    # set models to eval mode if we are doing testing
    if (test_mode):
        basic_ext.eval()
        shared_gen.eval()
        specific_gen.eval()
        print("load models in eval mode")

    models = {
        "basic_feat_ext": basic_ext,
        "shared_feat_gen": shared_gen,
        "specific_feat_gen": specific_gen
    }
    return models