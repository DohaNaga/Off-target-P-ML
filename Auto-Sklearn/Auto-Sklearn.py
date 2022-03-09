import autosklearn
import pickle
import autosklearn.classification
import numpy as np
import pandas as pd
import os
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import auc, plot_precision_recall_curve

#reading off target names from the main dataset

target_dataset = pd.read_excel("../Datasets/dataset_1.xlsx")
off_targets = target_dataset.OFF_TARGET.str.replace(" ","").unique()  

# function to calculate metrics
def calculate_metrics(y_test, y_predicted):
    """Calculate metrics for classification models"""

    data = {
        "Balanced Accuracy": [balanced_accuracy_score(y_test, y_predicted)],
        "Accuracy": [accuracy_score(y_test, y_predicted)],
        "Precision macro": [precision_score(y_test, y_predicted, average='macro')],
        "Average precision":[average_precision_score(y_test,y_predicted)],
        "Precision micro": [precision_score(y_test, y_predicted, average='micro')],
        "Precision weighted": [precision_score(y_test, y_predicted, average='weighted')],
        "Recall macro": [recall_score(y_test, y_predicted, average='macro')],
        "Recall micro": [recall_score(y_test, y_predicted, average='micro')],
        "Recall weighted": [recall_score(y_test, y_predicted, average='weighted')],
        "MCC": [matthews_corrcoef(y_test, y_predicted)],
        "Kappa": [cohen_kappa_score(y_test, y_predicted)],
        "F1 Score": [f1_score(y_test, y_predicted)],
        "F0.5 Score": [fbeta_score(y_test, y_predicted, beta=0.5)],
        "F2 Score": [fbeta_score(y_test, y_predicted, beta=2.0)],
    }
    df = pd.DataFrame(data)
    df.rename(index={0: "Metrics"}, inplace=True)
    return df


#calculating area under the roc and pr curve
def calculate_areas(y_test,y_proba):
    """Calculate roc auc and pr  for classification models"""
    precision, recall, thresholds  = precision_recall_curve(y_test,y_proba)
    data = {
        "AUC": [roc_auc_score(y_test,y_proba)],
        "AUCPR": [auc(recall, precision)],
     
    }
    df = pd.DataFrame(data)
    df.rename(index={0: "Metrics"}, inplace=True)
    return df



#configuration of the automl model
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task= 120, per_run_time_limit= 30,  resampling_strategy='cv',resampling_strategy_arguments={'folds':10},memory_limit=None, metric= autosklearn.metrics.balanced_accuracy,seed = 42)


#reading the train and test files of the datasets(which are similar to these of autogluon) and fitting the model for each dataset

for target in off_targets[0:len(off_targets)]:
    train_filename = f"Autogluon_files/train_{target}.csv"
    test_filename = f"Autogluon_files/test_{target}.csv"
    train_set =pd.read_csv(train_filename)
	test_set = pd.read_csv(test_filename)
	#adapting the dataframes for autosklearn
    X_train = train_set.drop(['ID','OFF_TARGET','BINARY_VALUE'], axis = 1)
    y_train = train_set['BINARY_VALUE']
    X_test = test_set.drop(['ID','OFF_TARGET','BINARY_VALUE'], axis = 1)
    y_test = test_set['BINARY_VALUE']
    
    #fitting the model
    automl.fit(X_train,y_train)
    
    #saving the model
    # save model
    with open(f"autosklearn_{target}.pkl", 'wb') as f:
        pickle.dump(automl, f)

    
    
    #taking a look on the models
    #print(automl.leaderboard())

    #predicting on the test set (values)
    y_pred= automl.predict(X_test)
    
    #predicting on the test set (probabilities) for the calculation of the auc and pr 
    y_proba = automl.predict_proba(X_test)
    
    #conf matrix
    conf_matrix= confusion_matrix(y_pred, y_test)
    
    
    #evaluation metrics and areas
    metrics_df_1 = calculate_metrics(y_test, y_pred).T
    metrics_df_2 = calculate_areas(y_test, y_proba[:,1]).T
    
    #appending both for a full metric df
    metrics_df = metrics_df_1.append(metrics_df_2)
    
    #writing down evaluation metrics and areas
    metrics_filename = f"/Auto-Sklearn/Metrics/metrics_{target}.csv"
    metrics_df.to_csv(metrics_filename)
    
    

