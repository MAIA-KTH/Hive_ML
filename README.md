# 4D_radiomics
A quick intro:
1) the files that are needed to be executed are placed in "run" folder.
2) there are 4 files in the run directory which will:
    2-1)  "1_analysis.py": general analysis of all the extracted dataset.
    2-2)  "2_SFS.py": sequential feature selection
    2-3)  "3_SFS_analysis.py" applying the cross validation analysis by using only the selected features
    2-4)  "4_SFS_analysis_prediction.py" applying the cross validation analysis on the training set and predicting the probability scores of the test set.
3) to run the codes, you can either execute them directly, or using the command line recalling with config file like: python 1_analysis.py -c ../configs/config.json
4) all the details of each executions will be saved as a json file in "log" directory.
5) performin the SFS method often takes a long time, the results will be stored in "selected_features" folder.
6) to extract the features, go to forlder "radiomics_extraction"
7) the extracted radiomics will be stored in "features" folder
8) the config file contains the following info that can be changed:
    8-1) path to extracted features in xlsx file format: up to 5 different feature set can be added which will be concatenated later.
    8-2) path to extracted features from the test set, this will be used only when working with ./run/4_SFS_analysis_prediction.py
    8-3) some parameters like seed for shuffling, n_fold in stratified cross val, number of stratified k-fold repeat, data normalization flag, and test_label flag: this one will            be used if we have the real labels of the test set in the corresponding xlsx file, for quantification.
    8_4) some model params: depth of DT,RF and Adab models, and number of estimators in DT, RF, and Adab models.
    8_5) number of features to be selected by SFS in a tuple like (3,4,5,...)
    8-6) "Analysis_After_SFS" these parameters will be used only when running ./run/3_SFS_analysis.py or ./run/4_SFS_analysis_prediction.py
          define the number of features to be recalled from the log files
          define the name of the learning algorithm that was used to extract the selected featurs
          define the name of the log file of feature selection to be recalled.
   8-7) "Learning_Algorithm" section: here you set the boolean flag for the model to be used for the analysis.
