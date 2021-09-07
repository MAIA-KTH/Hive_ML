import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LassoCV
from ReliefF import ReliefF


#### Removing Constant Features using Variance Threshold

def nonconstants(thr, x_train, x_test):
    '''
    thr=0,
    Constant features have zero variance. Threshold parameter
    is the variance.
    '''
    non_constant = []
    constant_filter = VarianceThreshold(threshold = thr)
    constant_filter.fit(x_train)
    nonconstant_index = constant_filter.get_support(indices = True)
    nonconstant_train = constant_filter.transform(x_train)
    nonconstant_test = constant_filter.transform(x_test)
    non_constant.append(nonconstant_train)
    non_constant.append(nonconstant_test)
    
    return non_constant


#### Removing Quasi-Constant Features using Variance Threshold

def nonquasi_constants(thr, x_train, x_test):
    '''
    thr=0.01
    Removing the features with 99% of similarity; Threshold (variance)
    parameter is set to 0.01
    '''
    nonq_constant = []
    qcontant_filter = VarianceThreshold(threshold = thr)
    qcontant_filter.fit(x_train)
    nonqconstant_index = qcontant_filter.get_support(indices = True)
    nonqconstant_train = qcontant_filter.transform(x_train)
    nonqconstant_test = qcontant_filter.transform(x_test)
    nonq_constant.append(nonqconstant_train)
    nonq_constant.append(nonqconstant_test)
    
    return nonq_constant


#### Removing Correlated Features

def uncorrelateds(corr_thresh, x_train, x_test):
    """
    Calculate the correlation matrix between the features
    and remove those with correlation higher than thresholds.
    corr_thresh = 0.8 (e.g.,)
    """
    uncorrelated = []
    correlated_ind = []
    uncorrelated_ind = []
    df_x_train = pd.DataFrame(x_train)
    correlation_matrix = df_x_train.corr()

    for ind in range(correlation_matrix.shape[1]):
        for ix in range(ind):
            if abs(correlation_matrix.iloc[ind, ix]) > corr_thresh:
                colname = correlation_matrix.columns[ind]
                correlated_ind.append(colname)
            else:
                column_name = correlation_matrix.columns[ind]
                uncorrelated_ind.append(column_name)
                    
                
    correlated_ind = list(set(correlated_ind))
    uncorr_x_train = np.delete(x_train, correlated_ind, axis = 1)
    uncorr_x_test = np.delete(x_test, correlated_ind, axis = 1)
    uncorrelated.append(uncorr_x_train)
    uncorrelated.append(uncorr_x_test)
    
    return uncorrelated


#### Least Absolute Shrinkage and Selection Operator (LASSO) 
#### Lasso regression for feature selection
"""
L1 regularization as penalty.
"""
# We use the base estimator LassoCV

def lasso(n_fold, max_iters, thr, x_train, y_train, x_test):
    """
    n_fold=5, max_iters=5000, thr = 0.6
    """
    lasso = []
    cv_lasso = LassoCV(cv = n_fold, max_iter = max_iters, n_jobs = 1)
    cv_lasso_model = SelectFromModel(cv_lasso, threshold = thr)
    cv_lasso_model.fit(x_train, y_train)
    #n_remained_lasso = cv_lasso_model.transform(x_train).shape[1]
    remained_lasso_idx = cv_lasso_model.get_support(indices = True)
    x_train_lasso = x_train[:,remained_lasso_idx] 
    x_test_lasso = x_test[:,remained_lasso_idx]
    lasso.append(x_train_lasso)
    lasso.append(x_test_lasso)
    
    return lasso


#### Relevance in Estimatory Features (RELIEF)

def relief(n_neighbors, n_features, x_train, y_train, x_test):
    """
    n_neighbors=1, n_features=20
    """
    relief_features = []
    relief = ReliefF(n_neighbors = n_neighbors, n_features_to_keep = n_features)
    relief.fit(x_train, y_train)
    relief_train = relief.transform(x_train)
    relief_test = relief.transform(x_test)
    relief_features.append(relief_train)
    relief_features.append(relief_test)
    
    return relief_features


#### Principle Component Analysis (PCA)

def pca(x_train, x_test):

    #pca_kernels = ('linear', 'poly', 'rbf', 'sigmoid', 'cosine')
    pca_kernels = ('linear', 'rbf')
    pca_features = {}
    for kernels in pca_kernels:
        
        if kernels == 'linear':
            for n_component in (10,5):
                features = []
                pca = PCA(n_components = n_component)
                pca.fit(x_train)
                x_train_pca = pca.transform(x_train)
                x_test_pca = pca.transform(x_test)
                features.append(x_train_pca)
                features.append(x_test_pca)
                pca_name = kernels+'_n_component_'+str(n_component)
                pca_features[pca_name] = features
                
        elif kernels == 'poly' or kernels == 'rbf' or \
        kernels ==  'sigmoid' or kernels == 'cosine':
            
            
            for n_component in (10,5):
                features = []
                pca = KernelPCA(n_components = n_component, kernel = kernels)
                pca.fit(x_train)
                x_train_pca = pca.transform(x_train)
                x_test_pca = pca.transform(x_test)
                pca_name = kernels+'_n_component_'+str(n_component)
                features.append(x_train_pca)
                features.append(x_test_pca)
                pca_features[pca_name] = features
    
    return pca_features


#### Mutual Information based  

def mutual_info(x_train, y_train, x_test):
    """
    Zero means independents and higher value presents higher dependency.
    neighbors:Number of neighbors to use for MI estimation for continuous variables
    """
    neighbors = [1, 2, 3, 4, 5]
    out_mi = {}
    for neighbor in neighbors:
        feature_mi = []
        mi_model = mutual_info_classif(x_train, y_train,
                                       n_neighbors = neighbor,
                                       copy = True)
        indpndts = np.where(mi_model==0)[0]
        mi_x_train = np.take(x_train, indpndts, axis=1)
        mi_x_test = np.take(x_test, indpndts, axis=1)
        feature_mi.append(mi_x_train)
        feature_mi.append(mi_x_test)
        out_mi[neighbor] = feature_mi
        
    return out_mi
        
  
#### Minimum Redundancy Maximum Relevance (MRMR)
"""
import pymrmr
df = pd.read_csv("A.csv")    
pymrmr.mRMR(df, 'MID', 100)
"""
    
"""
https://towardsdatascience.com/feature-selection-techniques-for-classification-and-python-tips-for-their-application-10c0ddd7918b    
"""





