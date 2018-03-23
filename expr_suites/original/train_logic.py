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

def _get_shared_code_loss(shared_feat_src,shared_feat_tgt,labels_src,labels_tgt,loss_coeff):
    "return a torch Variable as loss"
    total_loss = None
    # 1. cross-domain code similarity loss
    if (loss_coeff["cross"] > 0):
        cross_domain_loss = get_crossdom_pairwise_sim_loss(src_feats=shared_feat_src, tgt_feats=shared_feat_tgt,
                                                           src_labels=labels_src, tgt_labels=labels_tgt)
        total_loss = _add_loss_term(ori_loss=total_loss, add_term=loss_coeff["cross"] * cross_domain_loss)
    # 2. intr-source similarity loss
    if (loss_coeff["source"] > 0):
        intra_source_loss = get_pairwise_sim_loss(feats=shared_feat_src, labels=labels_src)
        total_loss = _add_loss_term(ori_loss=total_loss, add_term=loss_coeff["source"] * intra_source_loss)
    # 3. intra-target similarity loss
    if (loss_coeff["target"] > 0):
        intra_target_loss = get_pairwise_sim_loss(feats=shared_feat_tgt, labels=labels_tgt)
        total_loss = _add_loss_term(ori_loss=total_loss, add_term=loss_coeff["target"] * intra_target_loss)
    # 4. all-pair similarity loss
    if (loss_coeff["all"] > 0):
        all_pair_loss = get_pairwise_sim_loss(feats=torch.cat([shared_feat_src, shared_feat_tgt]),
                                              labels=torch.cat([labels_src,labels_tgt]))
        total_loss = _add_loss_term(ori_loss=total_loss, add_term=loss_coeff["all"] * all_pair_loss)

    # optionally add quantization loss
    if (loss_coeff["quantization"] > 0):
        final_feats = torch.cat([shared_feat_src, shared_feat_tgt])
        quantization_loss = _get_quantization_loss(continuous_code=final_feats)
        total_loss = _add_loss_term(ori_loss=total_loss,
                                    add_term=quantization_loss * loss_coeff["quantization"])
    return total_loss

def do_forward_pass(params, basic_feat_ext, shared_feat_gen, specific_feat_gen, imgs, labels):
    """
    :param params: should have property:
    (1) overall_loss_coeff = {shared:...,specific}
    (2) shared_loss_coeff = {cross:...,source:...,target:...,all:..,quantization:...}
    (3) specific_loss_coeff = {target:...,quantization}
    :param imgs: [src_imgs,tgt_imgs]
    :param labels: [src_labels,tgt_labels]
    :return: total loss
    """
    src_imgs, tgt_imgs = imgs
    labels_src,labels_tgt = labels
    basic_feat_src = basic_feat_ext(src_imgs)
    basic_feat_tgt = basic_feat_ext(tgt_imgs)
    total_loss = None
    specific_loss = 0
    shared_loss = 0

    # compute loss for shared code
    if (params.overall_loss_coeff["shared"] > 0):
        # generate shared feature
        shared_feat_src = shared_feat_gen(basic_feat_src)  # this should be outputs of tanh()
        shared_feat_tgt = shared_feat_gen(basic_feat_tgt)

        shared_loss = _get_shared_code_loss(shared_feat_src=shared_feat_src,shared_feat_tgt=shared_feat_tgt,
                                            labels_src=labels_src,labels_tgt=labels_tgt,loss_coeff=params.shared_loss_coeff)
        total_loss = _add_loss_term(ori_loss=total_loss,
                                    add_term=params.overall_loss_coeff["shared"] * shared_loss)

    # compute loss for specific code
    if (params.overall_loss_coeff["specific"] > 0):
        specific_feat_tgt = specific_feat_gen(basic_feat_tgt) # this should be outputs of tanh()
        specific_loss = _get_specific_code_loss(specific_feat=specific_feat_tgt,labels=labels_tgt,loss_coeff=params.specific_loss_coeff)
        total_loss = _add_loss_term(ori_loss=total_loss,add_term=params.overall_loss_coeff["specific"]*specific_loss)

    return {
        "total_loss": total_loss,
        "shared_loss": shared_loss * params.overall_loss_coeff["shared"],
        "specific_loss": specific_loss * params.overall_loss_coeff["specific"]
    }