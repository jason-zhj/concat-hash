"""
This is the main training procedure
"""
import itertools
import os
import torch
from torch.autograd import Variable

from commons.ml_train.utils import save_param_module, get_data_loader, save_models

def training(params, logger, train_func, model_def, save_model_to=""):
    # create data loader
    source_loader = get_data_loader(data_path=params.source_data_path,params=params)
    target_loader = get_data_loader(data_path=params.target_data_path,params=params)

    # model components
    basic_feat_ext = model_def.BasicFeatExtractor(params=params)
    shared_feat_gen = model_def.SharedFeatGen(params=params)
    specific_feat_gen = model_def.SpecificFeatGen(params=params)

    if (torch.cuda.is_available()):
        basic_feat_ext.cuda()
        shared_feat_gen.cuda()
        specific_feat_gen.cuda()

    # optimizers
    learning_rate = params.learning_rate
    opt_basic_feat = torch.optim.Adam(basic_feat_ext.parameters(), lr=learning_rate)
    opt_shared_feat = torch.optim.Adam(shared_feat_gen.parameters(), lr=learning_rate)
    opt_specific_feat = torch.optim.Adam(specific_feat_gen.parameters(), lr=learning_rate)

    # for saving gradient
    grad_records = {}
    def save_grad(name):
        def hook(grad):
            grad_records[name] = grad
        return hook

    # training
    total_loss_records = []
    shared_loss_records = []
    specific_loss_records = []

    for i in range(params.iterations):
        # refresh data loader
        itertools.tee(source_loader)
        itertools.tee(target_loader)
        zipped_loader = enumerate(zip(source_loader, target_loader))
        acc_total_loss, acc_shared_loss, acc_specific_loss = 0,0,0
        logger.info("epoch {}/{} started".format(i,params.iterations))
        # train using minibatches
        for step, ((images_src, labels_src), (images_tgt, labels_tgt)) in zipped_loader:
            logger.info("batch {}".format(step))
            # check GPU availability
            if torch.cuda.is_available():
                images_src = images_src.cuda()
                images_tgt = images_tgt.cuda()
                labels_src = labels_src.cuda()
                labels_tgt = labels_tgt.cuda()

            # clear gradients
            opt_basic_feat.zero_grad()
            opt_shared_feat.zero_grad()
            opt_specific_feat.zero_grad()

            # create input variables
            src_imgs = Variable(images_src).float()
            tgt_imgs = Variable(images_tgt).float()
            # src_lbls = Variable(labels_src)
            # tgt_lbls = Variable(labels_tgt)

            # call the specific training procedure to calculate loss
            train_out = train_func(params=params,basic_feat_ext=basic_feat_ext,shared_feat_gen=shared_feat_gen,specific_feat_gen=specific_feat_gen,
                       imgs=[src_imgs,tgt_imgs],labels=[labels_src,labels_tgt])
            total_loss,shared_loss,specific_loss = train_out["total_loss"],train_out["shared_loss"],train_out["specific_loss"]


            # do weights update
            if (total_loss is not None):
                acc_total_loss += total_loss.cpu().data.numpy()[0]

                total_loss.backward()
                opt_specific_feat.step()
                opt_basic_feat.step()
                opt_shared_feat.step()

            if (shared_loss !=0):
                acc_shared_loss += shared_loss.cpu().data.numpy()[0]
            if (specific_loss !=0):
                acc_specific_loss += specific_loss.cpu().data.numpy()[0]

        total_loss_records.append(acc_total_loss/(step+1))
        shared_loss_records.append(acc_shared_loss/(step+1))
        specific_loss_records.append(acc_specific_loss/(step+1))
        logger.info("epoch {} | total loss: {}, shared loss: {}, specific loss: {}".format(i, acc_total_loss / (step+1),acc_shared_loss/(step+1), acc_specific_loss/(step+1)))


    # save model params
    save_models(models={"basic_feat_ext":basic_feat_ext,"shared_feat_gen":shared_feat_gen,"specific_feat_gen":specific_feat_gen},
                save_model_to=save_model_to,save_obj=True,save_params=True)
    # save the training settings
    save_param_module(params=params,save_to=os.path.join(save_model_to, "train_settings.txt"))
    logger.info("model saved")

    return {
        "total_loss_records": total_loss_records,
        "specific_loss_records": specific_loss_records,
        "shared_loss_records": shared_loss_records
    }