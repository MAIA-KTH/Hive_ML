import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

def feature_set_details(feature_set):
    feature_list = []
    for sequence,features in feature_set.items():
        features_names = list(features.keys()) # == list(features_df.column)
        features_names = features_names[3:]       # getting the feature names
        subject_ids =  list(features[features.columns[1]]) # get the subject ids
        subject_label = np.asarray(list(features[features.columns[2]])) # get the subject labels
        feature_values = features.values  # get the feature values
        feature_values = feature_values[:,3:] # the first 3 columns contain order, id, labels 
        feature_values = feature_values.astype(np.float32)
        feature_list.append(feature_values)
    
    return feature_list, subject_ids, subject_label
    

def feature_squeeze(feature_list):
    
    feature_arrays = np.array(feature_list)
    mean_features = np.mean(feature_arrays, axis=0)
    sum_features = np.sum(feature_arrays, axis=0)
    std_features = np.std(feature_arrays, axis=0)
    
    delta_features = np.absolute(np.subtract(feature_arrays, mean_features))
    feature_set_squeeze = np.mean(delta_features, axis=0)
    
    return mean_features, sum_features, std_features, feature_set_squeeze


def data_shuffling(feature_set, label_set, seed_val):

    
    length = np.arange(label_set.shape[0])
    np.random.seed(seed_val)  
    np.random.shuffle(length)
    feature_set = feature_set[length]
    label_set = label_set[length]
    
    return feature_set, label_set


def feature_normalization(x_train, x_val = None, x_test = None):
    '''
    Normalize the each feature into the range of 0 to 1
    Parameters
    ----------
    x_train : array
        Feature matrix of training set.
    x_val : array
        Feature matrix of validation set.
    x_test : array, optional
        Feature matrix of test set. The default is None.
    Returns
    -------
    x_train, x_val, x_test : arrays
        normalized feature sets based on the statistics of training features.
    '''

    min_max_norm = preprocessing.MinMaxScaler(feature_range=(0,1))
    min_max_norm.fit(x_train)
    x_train = min_max_norm.fit_transform(x_train)
    if x_val is not None:
        x_val = min_max_norm.transform(x_val)
    
    if x_test is not None:
        x_test = min_max_norm.transform(x_test)
        
    return x_train, x_val, x_test


def get_stats(metric):
    '''
    Get mean and std of metrics
    Parameters
    ----------
    metric : list
        containing the metrics of different folds.
    Returns
    -------
    mean_metric : float
        average value of the metric.
    std_metric : float
        standard deviation of the metric.
    '''
    
    metric = np.array(metric)
    mean_metric = np.mean(metric)
    std_metric = np.std(metric)
    
    return mean_metric, std_metric

def kfold_stats(performance_summary):
    '''
    mean/std of metrics for cross validation experiments
    Parameters
    ----------
    performance_summary : dict
        evaluation metrics for each folds are kept in the dict.
    Returns
    -------
    four average values and four standard deviation values for four metrics 
    including accuracy, sensitivity, specifity, and auroc.
    '''

    acc = []
    sen = []
    spc = []
    auc = []
    auc.append(list(performance_summary['AUC_Val'].values()))
    acc.append(list(performance_summary['ACC_Val'].values()))
    sen.append(list(performance_summary['SEN_Val'].values()))
    spc.append(list(performance_summary['SPC_Val'].values()))
        
    acc = np.array(acc)
    sen = np.array(sen)
    spc = np.array(spc)
    auc = np.array(auc)
    
    mean_acc, std_acc = get_stats(acc)
    #mean_sen, std_sen = get_stats(sen)
    #mean_spc, std_spc = get_stats(spc)
    mean_auc, std_auc = get_stats(auc)
    
    return mean_acc, mean_auc, std_acc, std_auc, np.max(acc), np.max(auc)


def write_csv_predicton(performance_summary, experiment_name, subject_ids_unseen):
    
    testset_df = pd.DataFrame.from_dict(performance_summary['Prob_Test'])
    score_values = testset_df.values
    mean_predictions = np.mean(score_values, axis=1)
    testset_df.insert(0, "Mean_Prob", mean_predictions)
    testset_df.insert(0, "Subject_ID", subject_ids_unseen)
    csv_name = 'Test_Score_'+experiment_name+'.csv'
    csv_path = os.path.join('../logs/', csv_name)
    testset_df.to_csv(csv_path)



