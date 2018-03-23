import os
import numpy as np
from shutil import copyfile

def _block_copy(source_dir,filenames,target_dir):
    "copy `filenames` from source_dir to target_dir"
    for file in filenames:
        source_file = os.path.join(source_dir, file)
        tgt_file = os.path.join(target_dir, file)
        copyfile(source_file, tgt_file)

source_dir = "F:/data/domain_adaptation_images/original/webcam"
train_dir = "F:/data/domain_adaptation_images/train/webcam"
test_dir = "F:/data/domain_adaptation_images/test/webcam"

train_num = 3
test_num = 12

for subdir in os.listdir(source_dir):
    fns = os.listdir(os.path.join(source_dir,subdir))
    np.random.shuffle(fns)
    train_files = fns[:train_num]
    rest = fns[train_num:]

    # copy train files
    target_train_dir = os.path.join(train_dir,subdir)
    if (not os.path.exists(target_train_dir)): os.makedirs(target_train_dir)
    _block_copy(source_dir=os.path.join(source_dir,subdir),target_dir=target_train_dir,filenames=train_files)

    test_files = rest[:test_num] if len(rest) >= test_num else rest
    # copy test files
    target_test_dir = os.path.join(test_dir,subdir)
    if (not os.path.exists(target_test_dir)): os.makedirs(target_test_dir)
    _block_copy(source_dir=os.path.join(source_dir, subdir), target_dir=target_test_dir, filenames=test_files)

def list_file_counts(dir):
    dir = "F:/data/domain_adaptation_images/original/webcam"
    for subdir in os.listdir(dir):
        print(len(os.listdir(os.path.join(dir,subdir))))

