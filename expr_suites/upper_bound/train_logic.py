"""
specific training procedures other than the general training framework
"""
import torch
from commons.ml_train.utils import _add_loss_term,_get_quantization_loss, get_crossdom_pairwise_sim_loss,get_pairwise_sim_loss

def _get_specific_code_loss(specific_feat,labels,loss_coeff):
    "return a torch Variable as loss"
    total_loss = None
    if (loss_coeff["target"]>0):
        hash_loss = get_pairwise_sim_loss(feats=specific_feat,labels=labels)
        total_loss = _add_loss_term(ori_loss=total_loss,add_term=loss_coeff["target"] * hash_loss)

    if (loss_coeff["quantization"] > 0):
        quantization_loss = _get_quantization_loss(continuous_code=specific_feat)
        total_loss = _add_loss_term(ori_loss=total_loss,
                                    add_term=quantization_loss * loss_coeff["quantization"])
    return total_loss

def do_forward_pass(params, basic_feat_ext, specific_feat_gen, imgs, labels):
    """
    :param params: should have property:
    (1) overall_loss_coeff = {shared:...,specific}
    (2) shared_loss_coeff = {cross:...,source:...,target:...,all:..,quantization:...}
    (3) specific_loss_coeff = {target:...,quantization}
    :param imgs: [src_imgs,tgt_imgs]
    :param labels: [src_labels,tgt_labels]
    :return: total loss
    """
    basic_feat = basic_feat_ext(imgs)
    # compute loss for shared code
    specific_feat_tgt = specific_feat_gen(basic_feat) # this should be outputs of tanh()
    specific_loss = _get_specific_code_loss(specific_feat=specific_feat_tgt,labels=labels,loss_coeff=params.specific_loss_coeff)

    return {
        "total_loss": specific_loss,
    }