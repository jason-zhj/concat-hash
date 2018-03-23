import os
from models import params
from models import sinno_model as test_model
from ml_train.utils import LoggerGenerator, save_param_module
from training import training
from ml_test.testing import run_simple_test
from ml_toolkit.log_analyze import plot_trend_graph
"""
Train with shared + specific code
"""

# TODO: this changes from experiment to expr.
root_save_dir = "saved_models/check_var/together"
root_log_dir = "F:\\Project\\temp-projects\\pytorch-test\\log\\expr"

params.hash_size = 8 # TODO: this changes from experiment to expr.
params.specific_hash_size = 8
# 1. set the common params
#todo: temporarily swapped to train on source, but test on target
params.source_data_path = "F:/data/mnist/mini/training" #"F:/data/mnist/super-mini/training"
params.feat_gen_out_tanh = True
params.test_data_path = {
    "query": "F:/data/mnist_m/mini/query-db-split/query",
    "db": "F:/data/mnist_m/mini/query-db-split/db"
}
params.specific_loss_coeff = {
    "target": 1, "quantization": 0
}
params.shared_loss_coeff = {"all":0,"target":0.1,"quantization":0,"source":0.1,"cross":0.8}

params.use_specific_code = True
params.use_shared_code = True

save_param_module(params=params,save_to=os.path.join(root_save_dir, "common-params.txt"))

# 2. set the range for tunable params
overall_coeff_choices = [
    {"specific":0.5,"shared":0.5},
    {"specific":0.2,"shared":0.8},
    {"specific":0.8,"shared":0.2}
]



def train(params,id,overall_coeff):
    # 3. loop through the tunable choices and test
    params.overall_loss_coeff = overall_coeff
    # make the folders
    save_model_path = os.path.join(root_save_dir, str(id))
    save_result_path = os.path.join(save_model_path,"test_results")
    save_log_path = os.path.join(root_log_dir, "{}.txt".format(id))
    if (not os.path.exists(save_model_path)):
        os.makedirs(save_model_path)
        os.makedirs(save_result_path)

    # record the tuning params
    open(os.path.join(save_model_path,"params.txt"),"w").write("\n".join(["id",str(id),"overall_coeff",str(overall_coeff)]))
    print("start training for parameter set #{}".format(id))

    # perform training
    train_results = training(params=params, logger=LoggerGenerator.get_logger(
        log_file_path=save_log_path), save_model_to=save_model_path,model_def=test_model)
    # plot loss vs. iterations
    lines = [str(l) for l in train_results["total_loss_records"]]
    plot_trend_graph(var_names=["total loss"], var_indexes=[-1], var_types=["float"], var_colors=["r"], lines=lines,
                     title="total loss",save_to=os.path.join(save_result_path,"train-total_loss.png"),show_fig=False)

    with open(os.path.join(save_result_path,"train_records.txt"),"w") as f:
        f.write(str(train_results))
    print("finish training for parameter set #{}".format(id))

    # perform testing
    results = run_simple_test(params=params, saved_model_path=save_model_path,model_def=test_model)
    # save test results
    results["records"]["precision-recall-curve.jpg"].save(os.path.join(save_result_path,"precision-recall.png"))
    with open(os.path.join(save_result_path,"metrics.txt"),"w") as f:
        f.write(str(results["results"]))

    print("finish testing for parameter set #{}".format(id))


def test_all():
    target_ratios = [10,20,30,40]

    for i, ratio in enumerate(target_ratios):
        params.target_data_path = "F:/data/mnist_m/mini/train-{}-percent".format(ratio)
        for j,overall_coeff in enumerate(overall_coeff_choices):
            id = i * len(overall_coeff_choices) + j
            train(params=params,id=id,overall_coeff=overall_coeff)
            # do experiment on shared + specific

def try_best():
    params.iterations = 50
    overall_coeff = {"specific":0.2,"shared":0.8}
    for i in range(10):
        params.target_data_path = "F:/data/mnist_m/mini/temp/sample-{}".format(i)
        train(params=params, id=i, overall_coeff=overall_coeff)

if __name__ == "__main__":
    try_best()