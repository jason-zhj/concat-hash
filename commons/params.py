import platform
_IS_LINUX = platform.system() == "Linux"

########################################
# parameters for image preprocessing
########################################
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)

########################################
# parameters for the model
########################################
hash_size = 16  # hash size of shared hash code
specific_hash_size = 8 # size of specific code
use_dropout = False

########################################
# settings for training
########################################
train_shared_feat = True
train_specific_hash = True
only_tgt_specific_feat = False # only use SpecificFeatGen for target data
quantization_loss_coff = 1e-4
shared_feat_loss_coff = 1e-3
final_feat_loss = "hash" # classification or hash
hash_loss_only_for_src = False
# image-scale: 100 for office, 28 for mnist
image_scale = 28
shuffle_batch = True

## source data path
# mini-office: "F:/data/domain_adaptation_images/train-test-split-for-domadv/mini/train/source"
_windows_src_path = "F:/data/mnist/training"
_linux_src_path = "/home/zhangjie/data/mnist/train/source"
source_data_path = _linux_src_path if _IS_LINUX else _windows_src_path

## target data path
# mini-office: "F:/data/domain_adaptation_images/train-test-split-for-domadv/mini/train/target"
_windows_tgt_path = "F:/data/mnist_m/mini/train"
_linux_tgt_path = "/home/zhangjie/data/mnist/train/target"
target_data_path = _linux_tgt_path if _IS_LINUX else _windows_tgt_path

iterations = 50
batch_size = 30
learning_rate = 0.001 #TODO: tunable
save_model_path = "trained_models"

########################################
# settings for ml_test
########################################

# test data path
_windows_test_path = {
    "query": "F:/data/mnist_m/mini/test",
    "db": "F:/data/mnist/mini/training"
}
_linux_test_path = {
    "query": "",
    "db": ""
}
test_data_path = _linux_test_path if _IS_LINUX else _windows_test_path

test_batch_size = 100
# in ml_test, retrieval precision within hamming radius `precision_radius` will be calculated
precision_radius = 2