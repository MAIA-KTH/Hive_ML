import sys
sys.path.append('../../4D_radiomics/')
import numpy as np
from sklearn.metrics import roc_auc_score
from utilities.metrics_evaluatoin import conf_matrix, metrics




def learning(model,  x_train, y_train, x_val, y_val, x_test=None):
    '''
    ----------
    model : a SKlearn model
        can be RF, Adab, Tree etc.
    x_train : array
        training features
    y_train : array
        training labels.
    x_val : array
        validation features.
    y_val : array
        validation labels.
    x_test : array, optional
        test features don't have labels. 

    Returns
    -------
    roc_auc_val : float
        validation AUC.
    accuracy_val : float
        validatoin accuracy.
    sensitivity_val : float
        validation set sensitivity.
    specificity_val : float
        validation set specificity.
    y_probability_test : 
        test set prediction scores.

    '''
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    roc_auc_val = roc_auc_score(y_val, model.predict_proba(x_val)[:,1])
    
    conf_matrix_model = conf_matrix(y_val, y_pred)
    accuracy_val, sensitivity_val, specificity_val = metrics(conf_matrix_model)
    
    if x_test is not None:
        y_probability_test = model.predict_proba(x_test)[:,1]
    else:
        y_probability_test = None
    
    return roc_auc_val, accuracy_val, sensitivity_val, specificity_val, y_probability_test


def learning_with_sfs(sfs_model, feature_set, label_set, n_features=3):
    '''
    
    Parameters
    ----------
    sfs_model : class
        A compiled SFS model based on a sklearn algorithm.
    feature_set : array
        feature matrix of all data.
    label_set : array
        label vector representing the class labels.
    n_features : int
        number of features to be selected.
    Returns
    -------
    summary : dict
        returns a summary of the SFS model including the selected features,
        and performance of the model with the selected features.
    '''
    summary = {}
    sfs_fit = sfs_model.fit(feature_set, label_set)
    sfs_features = sfs_fit.subsets_  
    sfs_features = sfs_features[n_features]
    cv_scores = sfs_features['cv_scores']
    cv_average_score = np.mean(cv_scores)
    cv_std_score = np.std(cv_scores)
    
    selected_features = sfs_features['feature_idx']
    exp_name = str(n_features)+' selected features'
    metric_name_mean = exp_name+'_score_cv_mean'
    metric_name_std = exp_name+'_score_cv_std'
    metric_name_features = exp_name+'_names'
    summary[metric_name_mean] = cv_average_score
    summary[metric_name_std] = cv_std_score
    summary[metric_name_features] = selected_features
    
    return summary