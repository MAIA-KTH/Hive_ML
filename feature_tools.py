import numpy as np


def feature_set_details(feature_set):
    feature_list = []
    for sequence,features in feature_set.items():
        features_names = list(features.keys()) # == list(features_df.column)
        features_names = features_names[3:]       # getting the feature names
        subject_ids =  list(features[features.columns[1]]) # get the subject ids
        subject_label = np.asarray(list(features[features.columns[2]])) # get the subject labels
        feature_values = features.values  # get the feature values
        feature_values = feature_values[:,3:] # the first 3 columns contain order, id, lbels 
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


def normalize_Zscore(feature_set):
    set_mean = (np.vstack(np.mean(feature_set, axis=0))).T
    set_std = (np.vstack(np.std(feature_set, axis=0))).T
    set_normalized = (feature_set-set_mean)/set_std
    
    return set_mean, set_std, set_normalized

def get_stats(metric):
    
    metric = np.array(metric)
    mean_metric = np.mean(metric)
    std_metric = np.std(metric)
    
    return mean_metric, std_metric

def print_summary(cross_fold_dictionary):
    tmp_acc = []
    tmp_auc = []    
    for keys, values in cross_fold_dictionary.items():
        tmp_auc.append(values['roc_auc'])
        tmp_acc.append(values['accuracy'])
    auc_values = np.array(tmp_auc)
    acc_values = np.array(tmp_auc)

    print('avergae AUROC value is {a:1.3f} with std of {b:1.3f}'. \
          format(a=np.mean(auc_values), b=np.std(auc_values)))
    print('avergae ACC value is {a:1.3f} with std of {b:1.3f}'. \
          format(a=np.mean(acc_values), b=np.std(acc_values)))
    print('\n')

def print_summary_feature_selection(summary_file):
    for key, val in summary_file.items():
            temp_name = []
            temp_val = []
            for key_in, val_in in val.items():
                if val_in>=0:
                    temp_val.append(val_in)
                    temp_name.append(key_in)
                else:
                    pass
            mean_val = sum(temp_val)/len(temp_val)    
            max_val = max(temp_val)
            max_ind = temp_val.index(max_val)
            get_key = temp_name[max_ind]
            
            print('Method {} : Mean is {}, highest {} from seed {}'.format(
                key, mean_val,max_val, get_key))
    return max_val, get_key
            

def get_mean_prob(prob_dict):
    
    temp = np.array(list(prob_dict.values()))
    
    return  np.mean(temp, axis=0)