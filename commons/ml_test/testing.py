"""
This does hash-specific testing:
- precision within hamming radius
- precision vs. recall curve
"""
import torch, itertools, os
from ml_toolkit.data_process import batch_generator
from ml_toolkit.hash_toolkit.metrics.precision_recall import get_mean_avg_precision, calculate_precision_recall, get_precision_vs_recall, \
    plot_avg_precision_vs_recall
from ml_toolkit.hash_toolkit.metrics.ndcg import calculate_NDCG
from ml_toolkit.pytorch_utils.test_utils import load_models

from sklearn.model_selection import KFold  # import KFold
from torch import nn
from torch.autograd import Variable

from commons.ml_test.testing_utils import _get_data_loader, _construct_hash_function, _create_label_hash_dicts, _save_hash_code


# ml_test hashing performance
def testing(params,models,query_data_loader,db_data_loader,use_specific_code,use_shared_code,model_def):
    """
    :param models: a dict {name:model_obj}
    :param query_data_loader: a torch.utils.dataloader
    :param db_data_loader: a torch.utils.dataloader
    :return: return a dict {'results':[],'records':{filename:content}}
    """
    query_labels = []
    db_labels = []
    query_hash_ls = []
    db_hash_ls = []
    # 1. hash the whole db set and query set
    hash_model = _construct_hash_function(models=models,params=params,use_specific_code=use_specific_code,use_shared_code=use_shared_code,model_def=model_def)
    for i,(images,labels) in enumerate(db_data_loader):
        db_hash_ls += hash_model(images)
        db_labels += labels.numpy().tolist()
    for i, (images, labels) in enumerate(query_data_loader):
        query_hash_ls += hash_model(images)
        query_labels += labels.numpy().tolist()
    print("hashing finished")

    # 2. format data for ml_test
    db_set = _create_label_hash_dicts(hash_ls=db_hash_ls, label_ls=db_labels)
    query_set = _create_label_hash_dicts(hash_ls=query_hash_ls, label_ls=query_labels)

    # 3. do query for each data in `query_set`, compute precision, recall
    precision_recall_results = calculate_precision_recall(radius=params.precision_radius, db_set=db_set, test_set=query_set)
    print("finish calculating precision recalls")
    max_hdist = (params.hash_size if params.use_shared_code else 0) + (params.specific_hash_size if params.use_specific_code else 0)
    m_a_p = get_mean_avg_precision(test_set=query_set,db_set=db_set,maxdist=max_hdist)
    # ndcg_val = calculate_NDCG(query_set=query_set,db_set=db_set,radius_or_topk=50,use_topk=True)
    # plot precision vs. recall
    pr_dict = get_precision_vs_recall(test_set=query_set,db_set=db_set,max_hdist=max_hdist)
    precision_recall_dict = plot_avg_precision_vs_recall(pr_list=pr_dict,popup=False)
    # precision_recall_plot = None
    precision_recall_plot = precision_recall_dict["plot"]

    # 4. collect ml_test records
    test_results = {
        "precision-recall-results": precision_recall_results, # a dict
        "m_a_p":m_a_p,
        "precision-vs-recall":{
            "precisions": precision_recall_dict["avg_precisions"],
            "recalls": precision_recall_dict["avg_recalls"] # for later plotting precision-recall curve
        }
        # "ndcg-top50": ndcg_val
    }
    test_records = { #TODO: need further processing on this
        "precision-recall-curve.jpg": precision_recall_plot,
        "db_set": db_set,
        "query_set": query_set
    }
    return {
        "results":test_results,
        "records": test_records
    }


def testing_with_crossvalidation(params,saved_model_path,test_data_path,kfolds,model_def):
    "perform ml_test with cross validation by calling `ml_test()`"
    # load all ml_test data (shuffled)
    loader,fns = _get_data_loader(path=test_data_path, params=params, shuffle=True, use_batch=False)
    models = load_models(path=saved_model_path, model_names=["basic_ext","code_gen","abit_gen"])
    for data in loader:
        X = data[0]
        y = data[1]
    # divide data into k folds
    kf = KFold(n_splits=kfolds, shuffle=True)  # Define the split - into 2 folds
    all_results = []
    for db_index, query_index in kf.split(X):
        X_db, X_query = torch.FloatTensor(X.numpy()[db_index]), torch.FloatTensor(X.numpy()[query_index])
        y_db, y_query = torch.IntTensor(y.numpy()[db_index].tolist()), torch.IntTensor(y.numpy()[query_index].tolist())
        # prepare data loader
        query_loader = batch_generator(data=[X_query,y_query],batch_size=params.batch_size,shuffle=False)
        db_loader = batch_generator(data=[X_db, y_db], batch_size=params.batch_size, shuffle=False)
        results = testing(params=params,models=models,query_data_loader=query_loader,db_data_loader=db_loader,model_def=model_def,
                          use_shared_code=params.use_shared_code,use_specific_code=params.use_specific_code)
        all_results.append(results)

    return all_results

# this is a simple wrapper of `ml_test()`
def run_simple_test(params,saved_model_path, model_def, save_hash_to=None):
    "run a test with no cross validation, params should contain `use_specific_code`"
    # 1. load data
    query_loader, query_fns = _get_data_loader(path=params.test_data_path["query"], params=params,shuffle=False)
    db_loader, db_fns = _get_data_loader(path=params.test_data_path["db"], params=params,shuffle=False)
    # 2. load models
    models = _load_models_from_path(params=params, saved_model_path=saved_model_path,model_def_module=model_def)

    # 3. run ml_test
    test_results = testing(params=params, query_data_loader=query_loader, db_data_loader=db_loader,
                           models=models,use_specific_code=params.use_specific_code,use_shared_code=params.use_shared_code,model_def=model_def)
    # save hash code
    if (save_hash_to is not None):
        assert len(test_results["records"]["db_set"]) == len(db_fns)
        assert len(test_results["records"]["query_set"]) == len(query_fns)
        _save_hash_code(item_set=test_results["records"]["db_set"],fns=db_fns,
                        save_to=os.path.join(save_hash_to,"db.csv"))
        _save_hash_code(item_set=test_results["records"]["query_set"], fns=query_fns,
                        save_to=os.path.join(save_hash_to, "query.csv"))
    return test_results



if __name__ == "__main__":
    # run_simple_test(params=params,saved_model_path="trained_models/full_A")
    print("no main function")
