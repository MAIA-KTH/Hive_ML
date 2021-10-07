import json


config_data = {}

Feature_Train_Path = {}
Feature_Train_Path['feature_set1_path']="../features/batch1_class0.xlsx"
Feature_Train_Path['feature_set2_path']="../features/batch1_class1.xlsx"
Feature_Train_Path['feature_set3_path']="../features/batch2.xlsx"
Feature_Train_Path['feature_set4_path']=None
Feature_Train_Path['feature_set5_path']=None

Feature_Test_Path = {}
Feature_Test_Path['feature_set_test'] = "../features/batch2.xlsx"

Training_Params = {}
Training_Params['seed_val']=150
Training_Params['n_fold_split']=5
Training_Params['kf_repeat']=10
Training_Params['data_normalization']=True
Training_Params['test_labels_available']=False

Model_Params = {}
Model_Params['n_estimators']=100
Model_Params['depth_max']=10

Feature_Selection_Params = {}
Feature_Selection_Params['n_selecting_features']=(3,5,8,10,15,20,25,30)

Analysis_After_SFS = {}
Analysis_After_SFS['n_top_features']=10
Analysis_After_SFS['model_with_SFS']='rf'
Analysis_After_SFS['feature_selection_log_name']= 'Squeeze_set_SFS.json'

Learning_Algorithm = {}
Learning_Algorithm['rf_model']=True
Learning_Algorithm['dt_model']=False
Learning_Algorithm['adab_model']=False
Learning_Algorithm['lda_model']=False
Learning_Algorithm['qda_model']=False
Learning_Algorithm['naive_model']=False

config_data['Feature_Train_Path']=Feature_Train_Path
config_data['Feature_Test_Path']=Feature_Test_Path
config_data['Training_Params']=Training_Params
config_data['Model_Params']=Model_Params
config_data['Feature_Selection_Params']=Feature_Selection_Params
config_data['Analysis_After_SFS']=Analysis_After_SFS
config_data['Learning_Algorithm']=Learning_Algorithm


with open('config.json', 'w') as fp:
    json.dump(config_data, fp, indent = 4)