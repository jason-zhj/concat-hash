"""
Upper bound: use large number of target data to train, and test on target data
"""
import os, sys

sys.path.append(os.path.dirname(__file__))

from commons import params
import model as test_model
from train_logic import do_forward_pass
from commons.ml_train.utils import LoggerGenerator, save_param_module
from training import training
from testing import run_simple_test
from ml_toolkit.log_analyze import plot_trend_graph

# train on target, retrieve on target
root_save_dir = "saved_models/more-dropout"
root_log_dir = "log"

# train&test parameters
params.hash_size = 0 # hash size for shared code
params.specific_hash_size = 16

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

    print("finish testing for parameter set #{}".format(id))


if __name__ == "__main__":
    # train & test parameters
    params.target_data_path = "F:/data/mnist_m/mini/train"
    params.test_data_path = {
        "query": "F:/data/mnist_m/query-db-split/query",
        "db": "F:/data/mnist_m/query-db-split/db"
    }

    params.use_specific_code = True
    params.use_shared_code = False

    params.specific_loss_coeff = {
        "target": 1, "quantization": 0.05
    }
    params.iterations = 200

    train(params=params, id=0)