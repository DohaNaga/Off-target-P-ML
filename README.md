
# Off-targetPML

This repository contains the necessary scripts to build the binary off-target models explained in the paper from scratch using (1) A neural network framework (2)An automated machine learning framework (via Autogluon) and calculate the corresponding evaluation metrics and graphs for each model. 

It also contains the deep learning offtarget models (h5 format) constructed in the paper and the script needed to implement these models for any given structure.

A user can choose to :

(1) Develop new customed off-target models using the workflow
(2) predict the off-target profile read out (actives/inactives) for a set of given structures(Smiles)

## Support: doha.naga@roche.com


## (1) Develop customed off-target models using the workflow:
A sample of the main dataset used in the paper is provided : `dataset_1` which  consists of several coloumns, most importantly:

- COMPOUND_ID : Unique identifier of the compounds (in our case, Cas numbers are provided)
- OFF_TARGET : the name of the off-target against which the compound is screened
- SMILES
- BINARY_VALUE: whether the compound is active (1) or inactive (0) upon the corresponding target

##### Important note:

The sample data set `dataset_1` is provided for demonstration purposes. You can replace it with your own dataset (must have the same format and column annotations/names) to generate the prediction models for the desired targets. The necessary coloumns are the ones mentioned earlier (COMPOUND_ID,OFF_TARGET,SMILES,BINARY_VALUE ). The scripts provided are adapted to imbalanced datasets.

## I. Preparation of the working directory

- Download and place the folder ```off-targetmodels``` into your home directory

## II. Preparation of the models input for both the neuralnetworks and autogluon models

ECFP4 fingerprints are used for the predictions of the binary activities of the structures. These fingerprints need to be created as a first step and will be used as an input for the training of both: the neural networks and the autogluon models.

You will use the script `fingerprints_preparation.R` to generate the ECFP4 fingerprints for the compounds in `dataset_1`.

This script includes several curation steps :
- Removal of errored/incomplete smiles that were not parsed (warnings are generated at the end of script execution)
- Removal of intra-target duplicated smiles. 
- Removal of duplicated ids encoding same smiles.

The script is tested under R version 3.5.1 in R studio version 1.1.456.

##### Dependencies : 
- R 3.5.1
- rcdk 3.5.0
- fingerprint 3.5.7
- rcdklibs 2.3


##### Execution of the script
```sh
#navigate to the folder
$ cd Offtarget_models

#import the R version that will be used

ml R/3.5.1-goolf-1.7.20

#Run the script with the dataset file name as an argument
$ Rscript fingerprints_preparation.R dataset_1.xlsx

```

### Outcome
Three files are generated with the `Datasets` folder

- `dataset_2.csv` : A file contains the COMPOUND_ID of the molecules and their ECFP4 binary fingerprints.
- `dataset1_curated.xlsx` : Same as the input data (`dataset_1`), but after curation. This dataset is automatically used in the training
- `actives_inactives.xls` : A file contains the final number of actives and inactives for each target after curation

Curation: A curated dataset called `dataset1_curated.xlsx` is generated in the dataset folder and is automatically used in the training.

## III. Neural networks models
- The script is tested under R version 3.5.1 in R studio version 1.1.456.

- All scripts must be run from the NeuralNetworks directory

```sh
#navigate to the NeuralNetworks folder
$  cd NeuralNetworks
```


##### Dependencies : 
- Python ≥ 3.6
- reticulate 1.16
- Tensorflow 2.2.0
- Keras 2.3.0
- Tfruns 1.4


### 1- Installation

1. load python 3.6

  ```sh

#this line might vary on how your python versions are named

$ ml python/python3.6-2018.05 
  ```
3. Create a conda working environment from the unix command line and activate it

  ```sh

#creat conda environment using python 3.6

$ conda create -n r-tensorflow pip python = 3.6

#activate  environment

$ source activate r-tensorflow


  ```
3. Once env activated, install keras, tensorflow and tfruns from the environment

  ```sh
(r-tensorflow)$ pip install tensorflow

(r-tensorflow)$ pip install keras

(r-tensorflow)$ pip install tfruns
  ```

4. Testing if Tensoflow and Keras are successfuly installed

#Opening R version 3.5.1 from the terminal 


  ```sh
$  ml R/3.5.1-goolf-1.7.20 

  ```

5. loading keras and tensorflow libraries in R

  ```{r}  
use_condaenv("r-tensorflow")
library(keras)
library(tensorflow)
library(tfruns)

```

6. Testing if tensorflow is working in R 

```{r}  

mnist <- dataset_mnist()

x_train <- mnist$train$x

head(x_train)

#you should get back a matrix as follows
  [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13] [,14] [,15] [,16] [,17] [,18] [,19] [,20] [,21] [,22] [,23]

```

If you get an error regarding the locating python you can add in R:


```{r}
#path_python3 is the path to python3 in your conda env
use_python("path_python3")
```

For more information/problems regarding Tensorflow installation in R or alternative installation methods, please visit https://tensorflow.rstudio.com/installation/

### 2- Training

There are two main training scripts in the NeuralNetwork folder: 
- `tuning_1.R` creates the training/test sets, calls the script tuning_2.R and runs the grid search. 
- `tuning_2.R` creates, compiles and fits the models. 

##### Important notes:
- The grid search parameters used in the scripts are the same ones used in the paper, you can edit these parameters directly in `tuning_1.R`
- In `tuning_1.R` we save the runs with best evaluation accuracy, loss and balanced accuracy. The rest of the runs are cleaned and permanently deleted for memory issues. If you wish to do otherwise (e.g save all the runs), you can  edit directly in the script `tuning_1.R`

For more info on tfruns, please visit : https://tensorflow.rstudio.com/tools/tfruns/overview/

##### Execution of the training script 

- If you are running the script on your local machine:

```sh
#Execute the script tuning_1 (which calls and executes tuning_2.R)

$ Rscript tuning_1.R
 ```

- If you are running the script on a High Performance Cluster (HPC) machine with GPUS, you can use the `tuning.sh` script :

```sh
$ sbatch tuning.sh 
 ```

(Arguments of tuning.sh can be adjusted as required within the script)
 
### Outcome


A folder called `tuning` will be created. This folder should contain subfolders named by the OFF_TARGET name. These subfolders, will contain three folders:
- best_runs_ba : A folder containing the best model resulting from the grid search with respect to the best evaluation balanced accuracy.
- best_runs_acc :  A folder containing the best model resulting from the grid search with respect to the best evaluation  accuracy.
- best_runs_loss :  A folder containing the best model resulting from the grid search with respect to the best evaluation loss.
- grid_inforuns : A folder containing all the information on the grid search runs for the balanced accuracy for each target.

```
tuning
├──grid_inforuns
│  ├── 'OFF_TARGET'.xlsx
│
├── 'OFF_TARGET' best_runs_ba
│    ├──Run_YYYY_MM_DD-HH_MM_SS
│    │    ├──'OFF_TARGET'.h5 
│    │    ├── tfruns.d
│    │    │    ├──evaluation.json
│    │    │    ├──flags.json  
│    │    │    ├──metrics.json  
│    │    │
│    │    ├── plots
│    │    │      ├──Rplot001.png
│    │    │      
│    │    ├── 'OFF_TARGET'checkpoints
│    │  
├── 'OFF_TARGET'best_runs_acc
├── 'OFF_TARGET'best_runs_loss


```

### 3- Evaluation 

###### Dependencies
- caret 6.0-80
- yardstick 0.0.4
- PPROC 1.3.1
- ggpubr 0.2.3


The `evaluation.R` script :
- Imports the best model of each target (in terms of evalution balanced accuracy) in the .h5 format
- Evaluates it on the test sets (that werent used in the training or validation)
- Calculate the rest of the evaluation metrics(MCC,AUC,AUCPR,Accuracy) and draws AUC/AUCPR plots.

##### Execution of the evaluation script

```sh
$ Rscript evaluation.R
 ```
### Outcome

Within the folder `tuning`, the script creates an excel file `nn_bestba_allmetrics.xls` with the target name and corresponding evaluation metrics (of all target models) and a folder `plots` with the ROC and PR curves for all target models.

```
tuning
├──nn_bestba_allmetrics.xls
│  
├── plots
    ├── AUC
    │   ├── AUC_PLOT 'OFF_TARGET'.png
    │
    ├── AUCPR
        ├── PR_PLOT 'OFF_TARGET'.png
```



## IV. AutoGluon models

- The script `Autogluon_models.py` is tested under Python version 3.6.5 in Jupyter notebook  version 7.12.0. 
- The script constructs models for the 50 off-targets that are mentioned in the paper and are defined in a list in the begining of the script.
- The script must be run from the AutoGluon directory.


##### Dependencies : 
- Python ≥ 3.6
- MXNet ≥ 1.7.0.
- Autogluon 0.0.13
- sklearn 0.22.2
- numpy 1.19.2
- pandas 0.25.3
- xlrd 1.2.0
### 1- Installation

1- Use the same conda environment previously created for AutoGluon installation

```sh
$ source activate r-tensorflow
(r-tensorflow)$ python3 -m pip install -U setuptools wheel
(r-tensorflow)$ python3 -m pip install -U "mxnet<2.0.0, >=1.7.0"
(r-tensorflow)$ python3 -m pip install autogluon
  ```
For more information/problems or alternative installation methods for Autogluon installation, please visit  https://auto.gluon.ai/stable/install.html


### Input

The script `autogluon_fileprep.R` generates the training and test sets  in the required AutoGluon format. An example for this format is given in dummytrain_autogluon.csv where the coloumns are names in the following manner:

#the x1 to x1024 are the finger prints, the "ID" represents the compound ids, (in our case the  CAS.number), the BINARY_VALUE is the activity coloumn.



```sh
#navigate to the AutoGluon folder
$  cd Autogluon

```

```sh
#navigate to the AutoGluon folder
$  Rscript autogluon_filesprep.R

```
### Outcome

A folder called `Autogluon_files` will be produced. This folder contains the training and test sets for all the targets, named as follows `train_'OFF_TARGET'.csv` or `test_'OFF_TARGET'.csv`



### 2- Training and evaluation

- You can run the script  `Autogluon_models.py` within a jupyter notebook step by step or any other python interface for the training in the AutoGluon directory. This script trains the autogluon models for all the targets and evaluates them on the test sets as well.

- The training settings used in the scripts are the same used in the paper. For more information on other  training settings please visit https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.fit

##### Execution of the training/evaluation script 

```sh
python3 Autogluon_models.py
  ```


##### Outcome

Within the folder AutoGluon, a folder named `METRICS` will be created, this folder will contain csv files with the evaluation metrics of all the target models. These csv files will be named by the targetname.
For each target, a folder (named also by the target name) will be created. Each target folder will contain all the trained models and the fnal weighted ensemble model.

```
AutoGluon
├── METRICS
│   ├──metric_'OFF_TARGET'.csv
│
├── model_'OFF_TARGET'
    ├── models
        ├──model 1
        │  ├──model 1 fold 1
        │  │    ├── model.pkl
        │  ├──model 1 fold n  
        │       ├──model.pkl
        ├──model n
        │   ├──model  n fold 1
        │      ├──model.pkl     
        │   ├──model n fold n   
        │      ├──model.pkl
        │ 
        ├── weighted_ensemble
            ├──model.pkl

```



## (2) Predict the off-target profile read out (actives/inactives) for a set of input molecules(Smiles):

A sample of the input file is provided : `external_test.csv` which consists of two coloumns:

- COMPOUND_ID :Unique identifier of the compounds (in our case, Cas numbers are provided)
- SMILES

#### Important note :
Please make sure that : 
- Your input file has the same format and column annotations as `external_test.csv`.
- You have Tensorflow and Keras installed on your machine (please see above how to install)

##### Dependencies
The script is tested under R version 3.5.1 in R studio version 1.1.456.

Same dependencies as mentioned in Section II (preparation of fingerprints), III (neural networks) except for the tfruns (not needed).

##### Execution of the script
```sh
#navigate to the `Models` folder

$ cd Offtarget_models/Models

#Run the predictions with the input file as an argument
$  Rscript Off-targetP_ML.R external_test.csv

```

### Outcome

- predictions.xls : A matrix with the predictions (0:active, 1:inactive) for the input compounds vs the 50 targets. These predictions are generated using the roche in-house models.


