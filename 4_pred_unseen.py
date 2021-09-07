import warnings
warnings.filterwarnings("ignore")
import os
import copy
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from feature_tools import feature_set_details, feature_squeeze, get_stats, get_mean_prob
from learning_algorithms import clf_rf, clf_qda, clf_tree
from learning_algorithms import clf_lda, clf_adaboost, clf_naive


feature_set1 = pd.read_excel('features/feature_set_full1.xlsx', sheet_name=None) # reading training negative class 1
feature_set2 = pd.read_excel('features/feature_set_full2.xlsx', sheet_name=None) # reading training positive class 
feature_set_predict = pd.read_excel('features/feature_set_full_new_batch.xlsx', sheet_name=None) # reading test set
    
feature_list1, subject_ids1, subject_label1 = feature_set_details(feature_set1)
feature_list2, subject_ids2, subject_label2 = feature_set_details(feature_set2)
feature_list_predict, subject_ids_predict, _ = feature_set_details(feature_set_predict)

# time-dependent feature sets into single set
_, _, _, feature_set_squeeze1 = feature_squeeze(feature_list1)
_, _, _, feature_set_squeeze2 = feature_squeeze(feature_list2)
_, _, _, feature_set_squeeze_predict = feature_squeeze(feature_list_predict)


feature_set_squeeze = np.concatenate((feature_set_squeeze1, feature_set_squeeze2), axis=0)
feature_set_squeeze = feature_set_squeeze[:,14:] # the first 14 features are the same (geometric) so thery are equal to 0 after squeezing
subject_labels = np.concatenate((subject_label1,subject_label2))


feature_set_squeeze_predict = feature_set_squeeze_predict[:,14:]




# recalling the selected feature indices
selected_features_summary = pd.read_json('selected_features/Squeeze_set_SFS.json')
selected_features = selected_features_summary['rf']['features:5_feature_names']


# recalling the best seed
seed_summary = pd.read_json('summary_results/rf_sfs_5_features.json')
best_seed = seed_summary['best_seed_value'].values
best_seed = best_seed[0]


exp_name = 'TestSet_AUC_RF_Seed_'+str(best_seed)+'.csv'

feature_set_copy = copy.deepcopy(feature_set_squeeze)
subject_labels_copy = copy.deepcopy(subject_labels)
feature_set_pred_copy = copy.deepcopy(feature_set_squeeze_predict)

top_features = selected_features
feature_set = feature_set_copy[:, top_features]
feature_set_unseen = feature_set_pred_copy[:, top_features]
length = np.arange(subject_labels.shape[0])
   
np.random.seed(best_seed)  
np.random.shuffle(length)
feature_set = feature_set[length]
labels_set = subject_labels_copy[length]

# shuffling the orders
n_fold_split = 5


temp_rf_val = []
temp_knn_val = []
temp_adab_val = []
temp_lda_val = []
temp_qda_val = []
temp_naive_val = []
temp_dt_val = []
    
summary_rf_pred = {}
summary_knn_pred = {}
summary_adab_pred = {}
summary_lda_pred = {}
summary_qda_pred = {}
summary_naive_pred = {}
summary_dt_pred = {} 

seed_val = str(best_seed)
kf = KFold(n_splits = n_fold_split, shuffle = False) 
fold_num = 0
for train_index, val_index in kf.split(subject_labels):
    fold_num += 1
    fold_name = 'fold_'+str(fold_num)
    
    x_train = np.asarray(list(feature_set[i] for i in train_index))  
    x_val = np.asarray(list(feature_set[i] for i in val_index))
    y_train = np.asarray(list(labels_set[i] for i in train_index))
    y_val = np.asarray(list(labels_set[i] for i in val_index))
    
    x_test = feature_set_unseen
    
    min_max_norm = preprocessing.MinMaxScaler(feature_range=(0,1))
    min_max_norm.fit(x_train)
    x_train = min_max_norm.fit_transform(x_train, y_train)
    x_val = min_max_norm.transform(x_val)
    x_test = min_max_norm.transform(x_test)



    metrics_rf_val, prob_test_rf = clf_rf(None, x_train, y_train,  x_val, y_val, x_test)
    #auc_adab_val, prob_test_adab = clf_adaboost(4, 500, x_train, y_train, x_val, y_val, x_test)
    #auc_lda_val, prob_test_lda = clf_lda(x_train, y_train,  x_val, y_val, x_test)
    #auc_qda_val, prob_test_qda = clf_qda(x_train, y_train,  x_val, y_val, x_test)
    #auc_naive_val, prob_test_naive = clf_naive(x_train, y_train,  x_val, y_val, x_test)
    #auc_dt_val, prob_test_dt = clf_tree(None, x_train, y_train, x_val, y_val, x_test)
            
    
    temp_rf_val.append(metrics_rf_val['roc_auc'])
    #temp_adab_val.append(auc_adab)
    #temp_lda_val.append(auc_lda_val)
    #temp_qda_val.append(auc_qda_val)
    #temp_naive_val.append(auc_naive)
    #temp_dt_val.append(auc_dt_val)
    
    
    summary_rf_pred[fold_name] = prob_test_rf
    #summary_adab_pred[fold_name] = prob_test_adab
    #summary_lda_pred[fold_name] = prob_test_lda
    #summary_qda_pred[fold_name] = prob_test_qda
    #summary_naive_pred[fold_name] = prob_test_naive
    #summary_dt_pred[fold_name] = prob_test_dt


mean_prob = get_mean_prob(summary_rf_pred)
summary_rf_pred['mean'] = mean_prob
    
mean_val, std_val  = get_stats(temp_rf_val)
print('mean val is {} with std of {}'.format(mean_val, std_val))


df_summary = pd.DataFrame.from_dict(summary_rf_pred)
df_summary.insert(0, 'subject_id', subject_ids_predict)
save_path = os.path.join('prediction/', exp_name)      
df_summary.to_csv(save_path)
    
    