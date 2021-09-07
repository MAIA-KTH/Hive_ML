import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from feature_tools import feature_set_details, feature_squeeze, print_summary
from learning_algorithms import clf_rf, clf_qda
from learning_algorithms import clf_lda, clf_adaboost, clf_naive


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


# using the squeeze method
feature_set = feature_set_squeeze

# shuffling the orders
length = np.arange(subject_labels.shape[0])
np.random.seed(42)  
np.random.shuffle(length)
feature_set = feature_set[length]
subject_labels = subject_labels[length]



n_fold_split = 5


temp_rf = {}
temp_knn = {}
temp_adab = {}
temp_lda = {}
temp_qda = {}
temp_naive = {}



kf = KFold(n_splits = n_fold_split, random_state = 42, shuffle = True) 
fold_num = 0
for train_index, val_index in kf.split(subject_labels):
    fold_num += 1
    fold_name = 'fold_'+str(fold_num)
    print('Working on Fold Number:{}'.format(fold_num))
    
    x_train = np.asarray(list(feature_set[i] for i in train_index))  
    x_val = np.asarray(list(feature_set[i] for i in val_index))
    y_train = np.asarray(list(subject_labels[i] for i in train_index))
    y_val = np.asarray(list(subject_labels[i] for i in val_index))
    
    min_max_norm = preprocessing.MinMaxScaler(feature_range=(0,1))
    min_max_norm.fit(x_train)
    x_train = min_max_norm.fit_transform(x_train, y_train)
    x_val = min_max_norm.transform(x_val)


    rf_merics, _ = clf_rf(None, x_train, y_train, x_val, y_val, x_test=None)
    adab_metrics, _ = clf_adaboost(5, 550, x_train, y_train, x_val, y_val, x_test=None)
    lda_metrics, _ = clf_lda(x_train, y_train, x_val, y_val, x_test=None)
    qda_metrics, _ = clf_qda(x_train, y_train, x_val, y_val, x_test=None)
    naive_metrics, _ = clf_naive(x_train, y_train, x_val, y_val, x_test=None)
    
    temp_rf[fold_name] = rf_merics
    temp_adab[fold_name] = adab_metrics
    temp_lda[fold_name] = lda_metrics
    temp_qda[fold_name] = qda_metrics
    temp_naive[fold_name] = naive_metrics

print('\n'*3)
print('Random Forest results summary is:')
print_summary(temp_rf)
print('Adaboost results summary is:')
print_summary(temp_adab)
print('Linear Analysis results summary is:')
print_summary(temp_lda)
print('Quadratic Analysis results summary is:')
print_summary(temp_qda)
print('Naive Analysis results summary is:')
print_summary(temp_naive)





    
    