"""
Settings: for training, use 3 examples per category for DSLR and Webcam
use the rest for testing
"""
import os

from ml_toolkit.log_analyze import plot_trend_graph

from commons.ml_test.testing import run_simple_test
from expr_suites.use_office.train_logic import do_forward_pass
from expr_suites.use_office.training import training
from expr_suites.use_office import params
from commons.ml_train.utils import get_data_loader, save_param_module, LoggerGenerator
from expr_suites.use_office import model as test_model

root_save_dir = "saved_models/a2w"
root_log_dir = "F:\\Project\\temp-projects\\pytorch-test\\log\\expr"

params.hash_size = 8
params.specific_hash_size = 8
# 1. set the common params
params.source_data_path = "F:/data/domain_adaptation_images/original/amazon"
params.feat_gen_out_tanh = True
params.test_data_path = {
    "query": "F:/data/domain_adaptation_images/test/query-db-split/webcam/query",
    "db": "F:/data/domain_adaptation_images/test/query-db-split/webcam/db"
}
params.specific_loss_coeff = {
    "target": 1, "quantization": 0.05
}
params.shared_loss_coeff = {"all":0,"target":0.1,"quantization":0.05,"source":0.1,"cross":0.8}

params.use_specific_code = True
params.use_shared_code = True

save_param_module(params=params,save_to=os.path.join(root_save_dir, "common-params.txt"))

def train(params,id,overall_coeff):
    # 3. loop through the tunable choices and test
    params.overall_loss_coeff = overall_coeff
    # make the folders
    save_model_path = os.path.join(root_save_dir, str(id))
    save_result_path = os.path.join(save_model_path,"test_results")
    save_log_path = os.path.join(root_log_dir, "{}.txt".format(id))
    """
    if (not os.path.exists(save_model_path)):
        os.makedirs(save_model_path)
        os.makedirs(save_result_path)

    # record the tuning params
    open(os.path.join(save_model_path,"params.txt"),"w").write("\n".join(["id",str(id),"overall_coeff",str(overall_coeff)]))
    print("start training for parameter set #{}".format(id))

    # perform training
    train_results = training(params=params, logger=LoggerGenerator.get_logger(
        log_file_path=save_log_path), save_model_to=save_model_path,model_def=test_model,train_func=do_forward_pass)
    # plot loss vs. iterations
    lines = [str(l) for l in train_results["total_loss_records"]]
    plot_trend_graph(var_names=["total loss"], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title="total loss",save_to=os.path.join(save_result_path,"train-total_loss.png"),show_fig=False)
    
    with open(os.path.join(save_result_path,"train_records.txt"),"w") as f:
        f.write(str(train_results))
    print("finish training for parameter set #{}".format(id))
    """

    # perform testing
    results = run_simple_test(params=params, saved_model_path=save_model_path,model_def=test_model)
    # save test results
    results["records"]["precision-recall-curve.jpg"].save(os.path.join(save_result_path,"precision-recall.png"))
    with open(os.path.join(save_result_path,"metrics.txt"),"w") as f:
        f.write(str(results["results"]))

    print("finish testing for parameter set #{}".format(id))


if __name__ == "__main__":
    params.iterations = 150
    overall_coeff = {"specific": 0.2, "shared": 0.8}
    params.target_data_path = "F:/data/domain_adaptation_images/train/webcam"
    train(params=params, id=0, overall_coeff=overall_coeff)