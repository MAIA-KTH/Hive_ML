import sys
sys.path.append('../../4D_radiomics/')
import os
import json
from utilities.feature_tools import kfold_stats

DEFAULT_CONFIG_PATH = '../configs/config.json'

def summary_dictionary(temp_auc_val, temp_acc_val, temp_sen_val, temp_spc_val, temp_pred_test=None):   
    performance_summary = {}
    performance_summary['AUC_Val'] = temp_auc_val
    performance_summary['ACC_Val'] = temp_acc_val
    performance_summary['SEN_Val'] = temp_sen_val
    performance_summary['SPC_Val'] = temp_spc_val
    if temp_pred_test is not None:
        performance_summary['Prob_Test'] = temp_pred_test
    mean_acc, mean_auc, _, _, max_acc, max_auc= kfold_stats(performance_summary) 
    performance_statistic = {}
    performance_statistic['mean_AUC_Val'] = mean_auc
    performance_statistic['Max_AUC_Val'] = max_auc
    performance_statistic['mean_ACC_Val'] = mean_acc
    performance_statistic['Max_ACC_Val'] = max_acc
    performance_summary['Performance_Summary'] = performance_statistic  
    return performance_summary


def save_json(write_dir, filename, dict_summary):
    
    json_dir = os.path.join(write_dir, filename)
    with open(json_dir, 'w') as fp:
        json.dump(dict_summary, fp, indent = 4)
        
def load_json(json_path):
    if json_path:
        config_path = json_path
    else:
        config_path = DEFAULT_CONFIG_PATH
    with open(config_path, 'r') as json_file:
        configs  = json.load(json_file)
      
    return configs

def prepare_summary(feature_set1_path,feature_set2_path,
                                 feature_set3_path,feature_set4_path,
                                 feature_set5_path,feature_set_test,
                                 model_name, seed_val,n_fold_split,
                                 kf_repeat, data_normalization,
                                 n_estimators, depth_max,
                                 performance_summary):
    
    report_summary = {}
    report_summary['feature1_path'] = feature_set1_path
    report_summary['feature2_path'] = feature_set2_path
    report_summary['feature3_path'] = feature_set3_path
    report_summary['feature4_path'] = feature_set4_path
    report_summary['feature5_path'] = feature_set5_path
    report_summary['feature_test_path'] = feature_set_test
    report_summary['model_name_config'] = model_name
    report_summary['seed_value'] = seed_val
    report_summary['n_fold_cv'] = n_fold_split
    report_summary['n_repeat_KF'] = kf_repeat
    report_summary['feature_normalized'] = data_normalization
    report_summary['n_estimators'] = n_estimators
    report_summary['depth_max'] = depth_max
    report_summary['performance_summary'] = performance_summary
    
    return report_summary