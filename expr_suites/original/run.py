"""
Train with shared + specific code
"""

import os,sys

sys.path.append(os.path.dirname(__file__))

from commons import params
from commons.ml_train.utils import LoggerGenerator, save_param_module
from commons.ml_test.testing import run_simple_test
from ml_toolkit.log_analyze import plot_trend_graph

import model as test_model
from training import training
from train_logic import do_forward_pass

root_save_dir = "saved_models/together"
root_log_dir = "log"

def train(params,id):
    # make the folders
    save_model_path = os.path.join(root_save_dir, str(id))
    save_result_path = os.path.join(save_model_path,"test_results")
    save_log_path = os.path.join(root_log_dir, "{}.txt".format(id))
    if (not os.path.exists(save_model_path)):
        os.makedirs(save_model_path)
        os.makedirs(save_result_path)

    # perform training
    train_results = training(params=params, logger=LoggerGenerator.get_logger(
        log_file_path=save_log_path), save_model_to=save_model_path,model_def=test_model,train_func=do_forward_pass)

    # plot loss vs. iterations
    lines = [str(l) for l in train_results["total_loss_records"]]
    plot_trend_graph(var_names=["total loss"], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title="total loss",save_to=os.path.join(save_result_path,"train-total_loss.png"),show_fig=False)

    # perform testing
    results = run_simple_test(params=params, saved_model_path=save_model_path,model_def=test_model)

    # save test results
    results["records"]["precision-recall-curve.jpg"].save(os.path.join(save_result_path,"precision-recall.png"))
    with open(os.path.join(save_result_path,"metrics.txt"),"w") as f:
        f.write(str(results["results"]))



if __name__ == "__main__":
    # train & test settings
    params.hash_size = 8  # shared hash code size
    params.specific_hash_size = 8

    params.source_data_path = "F:/data/mnist/mini/training"
    params.target_data_path = "F:/data/mnist_m/mini/train-10-percent"
    params.test_data_path = {
        "query": "F:/data/mnist_m/mini/query-db-split/query",
        "db": "F:/data/mnist_m/mini/query-db-split/db"
    }

    params.specific_loss_coeff = {
        "target": 1, "quantization": 0
    }
    params.shared_loss_coeff = {"all": 0, "target": 0.1, "quantization": 0, "source": 0.1, "cross": 0.8}

    params.use_specific_code = True
    params.use_shared_code = True

    overall_coeff_choices = [
        {"specific": 0.5, "shared": 0.5},
        {"specific": 0.2, "shared": 0.8},
        {"specific": 0.8, "shared": 0.2}
    ]

    # try through different loss coefficients
    for j,overall_coeff in enumerate(overall_coeff_choices):
        params.overall_loss_coeff = overall_coeff
        train(params=params,id=j)