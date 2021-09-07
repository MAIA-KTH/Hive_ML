import warnings
warnings.filterwarnings('ignore')
import os
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
from feature_tools import feature_set_details, feature_squeeze
from learning_algorithms import dec_tree_model, ran_forest_model, quadratic_model, adaboost_model
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


experiment_name = 'Squeeze_set_SFS'
n_selecting_features = (3,4,5) # (3,5,8,10,15,20) # defining how many feature to be selected Time Consuming step
 
# loading the feature sets of two classes
feature_set1 = pd.read_excel('features/feature_set_full1.xlsx', sheet_name=None)
feature_set2 = pd.read_excel('features/feature_set_full2.xlsx', sheet_name=None)
    
feature_list1, subject_ids1, subject_label1 = feature_set_details(feature_set1)
feature_list2, subject_ids2, subject_label2 = feature_set_details(feature_set2)

#  time-dependent feature sets into single set
mean_features1, sum_features1, std_features1, feature_set_squeeze1 = feature_squeeze(feature_list1)
mean_features2, sum_features2, std_features2, feature_set_squeeze2 = feature_squeeze(feature_list2)

# concatenating two classes
sum_features = np.concatenate((sum_features1,sum_features2),  axis=0)
mean_features = np.concatenate((mean_features1,mean_features2),  axis=0)
feature_set_squeeze = np.concatenate((feature_set_squeeze1, feature_set_squeeze2), axis=0)
feature_set_squeeze = feature_set_squeeze[:,14:]
std_features = np.concatenate((std_features1, std_features2), axis=0)
std_features = std_features[:,14:]
subject_labels = np.concatenate((subject_label1,subject_label2))



# Feature Normalization
min_max_norm = preprocessing.MinMaxScaler()
feature_set = min_max_norm.fit_transform(feature_set_squeeze)



# shuffling the orders
length = np.arange(subject_labels.shape[0])
np.random.seed(42)  
np.random.shuffle(length)
feature_set = feature_set[length]
subject_labels = subject_labels[length]

# Compiling Learning Algorithms
dt_model = dec_tree_model()
rf_model = ran_forest_model()
adab_model = adaboost_model()
qd_model = quadratic_model()


all_algorithms = {}
all_algorithms['dt'] = dt_model
all_algorithms['rf'] = rf_model
all_algorithms['qda'] = adab_model
all_algorithms['adab'] = qd_model


# Sequential feature selection (SFS) for each of the lerning algorithm

all_results = {}
for key, value in all_algorithms.items():
    metrics = {}
    for n_features in n_selecting_features: 
        print('working on algorithm {} with {} features.'.format(key, n_features))
        sffs_model = SFS(value,
                         k_features = n_features,
                         forward = True,
                         floating = False,
                         scoring = 'roc_auc',
                         n_jobs = -1,
                         cv= 5)
    
        sffs = sffs_model.fit(feature_set, subject_labels)
        
        sffs_features = sffs.subsets_
        sffs_features = sffs_features[n_features]
        cv_scores = sffs_features['cv_scores']
        cv_average_score = np.mean(cv_scores)
        cv_std_score = np.std(cv_scores)
        feature_names = sffs_features['feature_idx']
        metric_name = 'features:'+str(n_features)
        metric_name_mean = metric_name+'_cv_mean'
        metric_name_std = metric_name+'_cv_std'
        metric_name_features = metric_name+'_feature_names'
        metrics[metric_name_mean] = cv_average_score
        metrics[metric_name_std] = cv_std_score
        metrics[metric_name_features] = feature_names

    all_results[key] = metrics
    

# Saving the params and results in a dictionary-json file.
jason_name = experiment_name+'.json'    
json_dir = os.path.join('selected_features/', jason_name)
with open(json_dir, 'w') as fp:
    json.dump(all_results, fp, indent = 4)
