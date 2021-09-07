import numpy as np
from sklearn import svm
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def conf_matrix(y_true, y_pred):
    target_labels = np.array(y_true)
    predictions = np.array(y_pred)
    matrix = confusion_matrix(target_labels, predictions)
    return matrix


def metrics(conf_matrix):
    
    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    
    accuracy = (float (tp+tn) / float(tp + tn + fp + fn))
    sensitivity = (tp / float(tp + fn))
    specificity = (tn / float(tn + fp))

    return accuracy, sensitivity, specificity


#### Decision Tree
def clf_tree(class_weight, x_train, y_train, x_val, y_val, x_test=None):
    metric_dic = {}
    criterion = 'gini' # 'gini' or 'entropy'
    max_depth = None # continues until all leaves are ...
    random_state = None
    dt_clf = DecisionTreeClassifier(criterion = criterion,
                                max_depth = max_depth,
                                class_weight = class_weight,
                                random_state = random_state)
    dt_clf.fit(x_train, y_train)
    y_pred_dt = dt_clf.predict(x_val)
    
    try:
        roc_auc = roc_auc_score(y_val, dt_clf.predict_proba(x_val)[:,1])
    except ValueError:
        roc_auc = -999
    conf_matrix_dt = conf_matrix(y_val, y_pred_dt)
    accuracy, sensitivity, specificity = metrics(conf_matrix_dt)
    metric_dic['accuracy'] = accuracy
    metric_dic['sensitivity'] = sensitivity
    metric_dic['specificity'] = specificity
    metric_dic['roc_auc'] = roc_auc 
    
    if x_test is not None:
        pred_unseen_prob = dt_clf.predict_proba(x_test)[:,1]
        y_probability_test = pred_unseen_prob
    else:
        y_probability_test = None
    
    return metric_dic, y_probability_test

#### Random Forest
def clf_rf(class_weight, x_train, y_train, x_val, y_val, x_test=None):
    metric_dic = {}
    criterion = 'gini' # 'gini' or 'entropy'
    max_depth = None 
    random_state = None
    rf_clf = RandomForestClassifier(criterion = criterion,
                                    max_depth = max_depth,
                                    class_weight = class_weight,
                                    random_state = random_state)
    rf_clf.fit(x_train, y_train)
    y_pred_rf = rf_clf.predict(x_val)
    y_pred_prob = rf_clf.predict_proba(x_val)[:,1]
    
    try:
        roc_auc = roc_auc_score(y_val, y_pred_prob)
    except ValueError:
        roc_auc = -999
    conf_matrix_rf = conf_matrix(y_val, y_pred_rf)
    accuracy, sensitivity, specificity = metrics(conf_matrix_rf)
    metric_dic['accuracy'] = accuracy
    metric_dic['sensitivity'] = sensitivity
    metric_dic['specificity'] = specificity
    metric_dic['roc_auc'] = roc_auc    
    
    if x_test is not None:
        pred_unseen_prob = rf_clf.predict_proba(x_test)[:,1]
        y_probability_test = pred_unseen_prob
    else:
        y_probability_test = None
    return metric_dic, y_probability_test



    
#### Adaptive Boosting
def clf_adaboost(max_depth, n_estimators, x_train, y_train, x_val, y_val, x_test=None):
    """
    max_depth = 8
    n_estimators = 500
    """
    metric_dic = {}
    adab_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth) ,
                                  n_estimators = n_estimators, random_state = None)
    adab_clf.fit(x_train, y_train)
    adab_y_pred = adab_clf.predict(x_val)
    y_pred_prob = adab_clf.predict_proba(x_val)[:,1]
    
    try:
        roc_auc = roc_auc_score(y_val, y_pred_prob)
    except ValueError:
        roc_auc = -999
    conf_matrix_adab = conf_matrix(y_val, adab_y_pred)
    accuracy, sensitivity, specificity = metrics(conf_matrix_adab)
    metric_dic['accuracy'] = accuracy
    metric_dic['sensitivity'] = sensitivity
    metric_dic['specificity'] = specificity
    metric_dic['roc_auc'] = roc_auc 
    
    if x_test is not None:
        pred_unseen_prob = adab_clf.predict_proba(x_test)[:,1]
        y_probability_test = pred_unseen_prob
    else:
        y_probability_test = None
    
    return metric_dic, y_probability_test

#### Discriminant analysis (linear and quadratic)
def clf_lda(x_train, y_train, x_val, y_val, x_test=None):
    
    metric_dic = {}
    lda_clf = LinearDiscriminantAnalysis()   
    lda_clf.fit(x_train, y_train)
    lda_y_pred = lda_clf.predict(x_val)
    
    try:
        roc_auc = roc_auc_score(y_val, lda_clf.predict_proba(x_val)[:,1])
    except ValueError:
        roc_auc = -999       
    conf_matrix_lda = conf_matrix(y_val, lda_y_pred)
    accuracy, sensitivity, specificity = metrics(conf_matrix_lda)
    metric_dic['accuracy'] = accuracy
    metric_dic['sensitivity'] = sensitivity
    metric_dic['specificity'] = specificity
    metric_dic['roc_auc'] = roc_auc 
    
    if x_test is not None:
        pred_unseen_prob = lda_clf.predict_proba(x_test)[:,1]
        y_probability_test = pred_unseen_prob
    else:
        y_probability_test = None
    
    return metric_dic, y_probability_test
    
def clf_qda(x_train, y_train, x_val, y_val, x_test=None):
    
    metric_dic = {}
    qda_clf =  QuadraticDiscriminantAnalysis()
    qda_clf.fit(x_train, y_train)
    qda_y_pred = qda_clf.predict(x_val)
    
    try:
        roc_auc = roc_auc_score(y_val, qda_clf.predict_proba(x_val)[:,1])
    except ValueError:
        roc_auc = -999
    conf_matrix_qda = conf_matrix(y_val, qda_y_pred)
    accuracy, sensitivity, specificity = metrics(conf_matrix_qda)
    metric_dic['accuracy'] = accuracy
    metric_dic['sensitivity'] = sensitivity
    metric_dic['specificity'] = specificity
    metric_dic['roc_auc'] = roc_auc 
    
    if x_test is not None:
        pred_unseen_prob = qda_clf.predict_proba(x_test)[:,1]
        y_probability_test = pred_unseen_prob
    else:
        y_probability_test = None
    
    return metric_dic, y_probability_test

#### Naive Bayes
def clf_naive(x_train, y_train, x_val, y_val, x_test=None):
    metric_dic = {}
    naive_clf = naive_bayes.GaussianNB()
    naive_clf.fit(x_train, y_train)
    naive_y_pred = naive_clf.predict(x_val)
    
    try:
        roc_auc = roc_auc_score(y_val, naive_clf.predict_proba(x_val)[:,1])
    except ValueError:
        roc_auc = -999
    conf_matrix_naive = conf_matrix(y_val, naive_y_pred)
    accuracy, sensitivity, specificity = metrics(conf_matrix_naive)
    metric_dic['accuracy'] = accuracy
    metric_dic['sensitivity'] = sensitivity
    metric_dic['specificity'] = specificity
    metric_dic['roc_auc'] = roc_auc
    
    if x_test is not None:
        pred_unseen_prob = naive_clf.predict_proba(x_test)[:,1]
        y_probability_test = pred_unseen_prob
    else:
        y_probability_test = None

    return metric_dic, y_probability_test



def dec_tree_model():

    dt_model = DecisionTreeClassifier(criterion = 'gini',
                                    max_depth = None,
                                    class_weight = None,
                                    random_state = None)
    return dt_model

def ran_forest_model():
    
    rf_model = RandomForestClassifier(criterion = 'gini',
                                max_depth = None,
                                class_weight = None,
                                random_state = None)
    return rf_model

def svm_linear_model():
    
    svm_model = svm.LinearSVC(penalty = 'l2', loss = 'squared_hinge',
                                    dual = True, tol = 1e-1,
                                    class_weight = None,
                                    random_state = None, C = 500,
                                    max_iter = -1)
    return svm_model

def knn_model():
    
    knnmodel = KNeighborsClassifier(n_neighbors = 3,
                                   weights = 'distance') 
    return knnmodel

def adaboost_model():
    
    adab_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 10) ,
                                  n_estimators = 500, random_state = None)
    return adab_model

def linear_model():   
    lda_model = LinearDiscriminantAnalysis()   
    return lda_model

def quadratic_model():
    
    qda_model =  QuadraticDiscriminantAnalysis()
    return qda_model

def naive_model():
    
    naiv_model = naive_bayes.GaussianNB()

    return naiv_model

    