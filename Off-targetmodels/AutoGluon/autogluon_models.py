

#importing necessary modules
import numpy as np
import pandas as pd
import os
import mxboard
import autogluon as ag
from autogluon import TabularPrediction as task
import sklearn


print(f'Autogluon version: {ag.__version__}')

#number of targets
#adapt n and off_targets according to the target models required
n = 50
off_targets = ['MONOAMINEOXIDASE-A',
  'PHOSPHODIESTERASE3B',
 'ACETYLCHOLINESTERASE',
 'ABL1',
 'PCPRECEPTOR',
  'CHOLECYSTOKININ1RECEPTOR',
 'SEROTONINTRANSPORTER',
 'DOPAMINED1RECEPTOR',
 'CANNABINOIDRECEPTORCB1',
 'GSK-3B',
 'PROSTAGLANDINFRECEPTOR',
 'NICOTINICRECEPTOR,NEURONAL(alpha-BGTXinsens.)',
 '5HT2A',
 'ZAP70',
 'HISTAMINEH1RECEPTOR',
 'GSK-3A',
 'HISTAMINEH2RECEPTOR',
 'CDK2',
 'PPARgamma',
 'ADENOSINEA3RECEPTOR',
 'XANTHINEOXIDASE',
 'ANDROGENRECEPTOR',
 'ANGIOTENSINCONVERTINGENZYME',
 '5HT2B',
 'MUSCARINICRECEPTORM1',
 'beta1-ADRENOCEPTOR',
 'ADENOSINEA1RECEPTOR',
 'CA2+CHANNEL(DILTIAZEMSITE)',
 'GLUCOCORTICOIDRECEPTOR',
 'PHOSPHODIESTERASE4D2',
 'NICOTINICRECEPTOR,MUSCLE',
 'NOREPINEPHRINETRANSPORTER',
 'ESTROGENRECEPTORalpha',
 'GLYCINERECEPTOR,STRYCHNINEINSENSITIVE',
 'ANGIOTENSINIIRECEPTOR1',
 'GABACL-CHANNELRECEPTOR',
 'alpha2A-ADRENOCEPTOR',
 'DOPAMINED2RECEPTOR(short)',
 '5HT3',
 'muOPIOIDRECEPTOR',
 'HIV-1PROTEASE',
 'CYCLOOXYGENASE2',
 'HISTAMINEH3RECEPTOR',
 'MMP-9',
 'alpha1A-ADRENOCEPTOR',
 'beta2-ADRENOCEPTOR',
 'GABA-A(BENZODIAZAPINEBINDINGSITE)',
 'kappaOPIOIDRECEPTOR',
 'MUSCARINICRECEPTORM2',
 '5HT1A']



# Metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


 
def calculate_metrics(y_test, y_predicted):
    """Calculate metrics for classification models"""

    data = {
        "Balanced Accuracy": [balanced_accuracy_score(y_test, y_predicted)],
        "Accuracy": [accuracy_score(y_test, y_predicted)],
        "Precision macro": [precision_score(y_test, y_predicted, average='macro')],
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



for target in off_targets[0:n]:
    train_filename = f"/Autogluon_files/train_{target}.csv"
    
    print(train_filename)
    #load data    
    train_data = task.Dataset(file_path=train_filename)
    train_data = train_data.loc[:,~train_data.columns.str.startswith('OFF_TARGET')]
    #prepare data
    
     
    label_column = 'BINARY_VALUE'
    
    output_directory = f"/model_{target}"
    presets = ['best_quality_with_high_quality_refit', 'optimize_for_deployment']
    #presets = ['best_quality_with_high_quality_refit']

    predictor = task.fit(
        train_data=train_data,
        label=label_column,
        id_columns=['ID'],
        output_directory=output_directory,
        presets=presets,
        problem_type='binary',
        eval_metric='balanced_accuracy',
        verbosity=1,
        random_seed = 42
    )

    test_filename = f"/Autogluon_files/test_{target}.csv"
    
    #print(train_filename)
    #load data and prepare   
    test_data = task.Dataset(file_path=test_filename)
    test_data = test_data.loc[:,~test_data.columns.str.startswith('OFF_TARGET')]
    
    
    # Test model
    y_test = test_data[label_column]
    test_data_without_y = test_data.drop(labels=[label_column], axis=1)  # delete label column
    y_pred = predictor.predict(test_data_without_y)
    metrics_df = calculate_metrics(y_test, y_pred).T
    metrics_filename = f"/METRICS/metrics_{target}.csv"
    metrics_df.to_csv(metrics_filename)
    



calculate_metrics(y_test, y_pred).T


train_files = glob.glob("/dir/*.csv")


#prepare data

#dropping un necessary coloumns
#train_data = train_data.loc[:,~train_data.columns.str.startswith('CEREP_TARGET')]

label_column = 'BINARY_VALUE'
print("Label column: \n", train_data[label_column].describe())



###defining output folder,presets and predictor

output_directory = f"/model_{target}.csv"
#presets = ['best_quality_with_high_quality_refit', 'optimize_for_deployment']
presets = ['best_quality_with_high_quality_refit']

predictor = task.fit(
        train_data=train_data,
        label=label_column,
        id_columns=['ID'],
        output_directory=output_directory,
        presets=presets,
        problem_type='binary',
        eval_metric='roc_auc',
        verbosity=1,
        visualizer='tensorboard',
)


results = predictor.fit_summary()



# Test model
y_test = test_data[label_column]
test_data_without_y = test_data.drop(labels=[label_column], axis=1)  # delete label column

y_score = predictor.predict_proba(test_data_without_y)
perf_score = predictor.evaluate_predictions(y_true=y_test, y_pred=y_score, auxiliary_metrics=True)
#print(perf_score)
print("ROC AUC: ", perf_score['roc_auc'].round(3))


y_pred = predictor.predict(test_data_without_y)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
pd.DataFrame(perf.values(), index=perf.keys(), columns=['value']).round(decimals=3)

calculate_metrics(y_test, y_pred).T

metrics_df = calculate_metrics(y_test, y_pred).T
metrics_df.to_csv("/METRICS/my_csv_file.csv")




import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test,  y_score)
auc = roc_auc_score(y_test, y_score)

plt.plot(fpr,tpr,color='darkorange',
         lw=2, label=f"ROC curve (area = {auc:.3f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


aucpr = average_precision_score(y_test, y_score)

