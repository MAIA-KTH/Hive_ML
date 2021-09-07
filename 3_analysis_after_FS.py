import warnings
warnings.filterwarnings("ignore")
import os
import copy
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from feature_tools import feature_set_details, feature_squeeze, get_stats, print_summary_feature_selection
from learning_algorithms import clf_rf, clf_qda, clf_tree
from learning_algorithms import clf_lda, clf_adaboost, clf_naive



feature_set1 = pd.read_excel('features/feature_set_full1.xlsx', sheet_name=None)
feature_set2 = pd.read_excel('features/feature_set_full2.xlsx', sheet_name=None)
    
feature_list1, subject_ids1, subject_label1 = feature_set_details(feature_set1)
feature_list2, subject_ids2, subject_label2 = feature_set_details(feature_set2)

#  time-dependent feature sets into single set
_, _, _, feature_set_squeeze1 = feature_squeeze(feature_list1)
_, _, _, feature_set_squeeze2 = feature_squeeze(feature_list2)



feature_set_squeeze = np.concatenate((feature_set_squeeze1, feature_set_squeeze2), axis=0)
feature_set_squeeze = feature_set_squeeze[:,14:]
subject_labels = np.concatenate((subject_label1,subject_label2))

main_set = feature_set_squeeze

# recalling the selected feature indices
selected_features_summary = pd.read_json('selected_features/Squeeze_set_SFS.json')
selected_features = selected_features_summary['rf']['features:5_feature_names']


len_features = len(selected_features)
experiment_name = 'rf_sfs_'+str(len_features)+'_features'

# set 100 random value as initial seeds
rand_seeds = np.random.randint(low=0, high=10000,
                               size=100)

# shuffling the orders
n_fold_split = 5


temp_rf_seed = {}
temp_adab_seed ={}
temp_lda_seed = {}
temp_qda_seed = {}
temp_naive_seed = {}
temp_dt_seed = {}
summary = {}

# for each of seed value, run a cross validation with only selected features
counter = 0
for seed_val in rand_seeds:
    counter += 1
    print('Working on seed:{}'.format(counter))
    temp_rf = []
    temp_adab = []
    temp_lda = []
    temp_qda = []
    temp_naive = []
    temp_dt = []
        
    feature_set_copy = copy.deepcopy(main_set)
    subject_labels_copy = copy.deepcopy(subject_labels)
    top_features = selected_features
    feature_set = feature_set_copy[:, top_features]
    length = np.arange(subject_labels.shape[0])
       
    np.random.seed(seed_val)  
    np.random.shuffle(length)
    feature_set = feature_set[length]
    labels_set = subject_labels_copy[length]
     
    seed_val = str(seed_val)
    kf = KFold(n_splits = n_fold_split, shuffle = False) 
    
    for train_index, val_index in kf.split(subject_labels):
        
        
        x_train = np.asarray(list(feature_set[i] for i in train_index))  
        x_val = np.asarray(list(feature_set[i] for i in val_index))
        y_train = np.asarray(list(labels_set[i] for i in train_index))
        y_val = np.asarray(list(labels_set[i] for i in val_index))
        
        min_max_norm = preprocessing.MinMaxScaler(feature_range=(0,1))
        min_max_norm.fit(x_train)
        x_train = min_max_norm.fit_transform(x_train, y_train)
        x_val = min_max_norm.transform(x_val)

    
    
        metrics_rf, _ = clf_rf(None, x_train, y_train, x_val, y_val, x_test=None)
        #metrics_adab, _ = clf_adaboost(4, 500, x_train, y_train, x_val, y_val, x_test=None)
        #metrics_lda, _ = clf_lda(x_train, y_train, x_val, y_val, x_test=None)
        #metrics_qda, _ = clf_qda(x_train, y_train, x_val, y_val, x_test=None)
        #metrics_naive, _ = clf_naive(x_train, y_train, x_val, y_val, x_test=None)
        #metrics_dt, _ = clf_tree(None, x_train, y_train, x_val, y_val, x_test=None)
                
        
        temp_rf.append(metrics_rf['roc_auc'])
        #temp_adab.append(metrics_adab['roc_auc'])
        #temp_lda.append(metrics_lda['roc_auc'])
        #temp_qda.append(metrics_qda['roc_auc'])
        #temp_naive.append(metrics_naive['roc_auc'])
        #temp_dt.append(metrics_dt['roc_auc'])
     
    mean_rf, std_rf  = get_stats(temp_rf)
    #mean_adab, std_adab  = get_stats(temp_adab)
    #mean_lda, std_lda  = get_stats(temp_lda)
    #mean_qda, std_qda  = get_stats(temp_qda)
    #mean_naive, std_naive  = get_stats(temp_naive)
    #mean_dt, std_dt  = get_stats(temp_dt)
    
    temp_rf_seed[seed_val] = mean_rf
    #temp_adab_seed[seed_val] = mean_adab
    #temp_lda_seed[seed_val] = mean_lda
    #temp_qda_seed[seed_val] = mean_qda
    #temp_naive_seed[seed_val] = mean_naive
    #temp_dt_seed[seed_val] = mean_dt
    

summary['rf'] = temp_rf_seed
#summary['adab'] = temp_adab_seed
#summary['lda'] = temp_lda_seed
#summary['qda'] = temp_qda_seed
#summary['naive'] = temp_naive_seed
#summary['dt'] = temp_dt_seed
highest_mean_AUC, best_seed = print_summary_feature_selection(summary)


summary['best_MeanAUC_observed'] = str(highest_mean_AUC)
summary['best_seed_value'] = best_seed
        
 # Saving the params and results in a dictionary-json file.    
json_name = experiment_name+'.json'
save_path_exp = os.path.join('summary_results/', json_name)
with open(save_path_exp, 'w') as fp:
    json.dump(summary, fp, indent = 4)   
  

    
    
    