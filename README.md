
# Off-target modelling
This repository contains the necessary scripts to derive off-target models using (1) A neural network framework (2)An autmomated machine learning framework (via Autogluon).

The main dataset is `dataset_1` which consists of several coloumns, most importantly:

- CAS.number : public id for the compounds
-  OFF_TARGET : the name of the off-target against which the compound is screened
-  SMILES
-  BINARY_VALUE: whether the compound is active (1) or inactive (0) upon the corresponding target

You can replace `Dataset_1` with your own dataset (must be the same name and structure).
## I. Preparation of the working directory

- Download and place the folder ```offtarget-models``` into your home directory

## II. Preparation of the models input for both the neuralnetworks and autogluon models

ECFP4 fingerprints are used for the predictions of the binary activities of the structures. These fingerprints need to be created as a first step and will be used as an input for the training of both: the neural networks and the autogluon models.

You will use the script `fingerprints_preparation.R` to generate the ECFP4 fingerprints for the compounds in `dataset_1`.

The script is tested under R version 3.5.1 in R studio version 1.1.456.

##### Dependencies : 
- R 3.5.1
- rcdk 3.5.0
- rcdklibs 2.3


##### Execution of the script
```sh
#navigate to the folder
$ cd offtarget_models

#Run the script
$ Rscript fingerprints_preparation.R

```

### Outcome

A file named `dataset_2` will be produced which contains the CAS.Number of the molecules and their ECFP4 binary fingerprints.

## III. Neural networks models
The script is tested under R version 3.5.1 in R studio version 1.1.456.

##### Dependencies : 
- Python ≥ 3.6
- reticulate 1.16
- Tensorflow 2.2.0
- Keras 2.3.0
- Tfruns 1.4


### 1- Installation


1. Create a conda working environment from the unix command line

  ```sh

#creat conda environment using python 3.6

$ conda create -n r-tensorflow pip python = 3.6

#activate  environment

$ source activate r-tensorflow


  ```
3. once env activated, install keras, tensorflow and tfruns from the environment

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

#should get back a matrix means th
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

##### Execution of the training script 

```sh
#navigate to the NeuralNetworks folder
$  cd NeuralNetworks

#Execute the script tuning_1 (which calls and executes tuning_2.R)

$ Rscript tuning_1.R
 ```

### Outcome
A folder called `tuning` will be created. This folder should contain subfolders named by the OFF_TARGET. These subfolders, will contain three folders:
- best_runs_ba : A folder containing the best model resulting from the grid search with respect to the validation balanced accuracy.
- best_runss_acc :  A folder containing the best model resulting from the grid search with respect to the validation  accuracy.
- best_runs_loss :  A folder containing the best model resulting from the grid search with respect to the validation loss.

Inside these subdirectories, the should be present in .h5 format and the validation .json files

Describe that you can use it for this specific target list or for another one

 describe the output files generated , the sh file 



### 3- Evaluation 
Describe the evaluation file



## IV. AutoGluon models
The script is tested under Python version 3.6.5 in Jupyter notebook  version 7.12.0. 

##### Dependencies : 
- Python ≥ 3.6
- MXNet ≥ 1.7.0.
- Autogluon 0.0.13
- sklearn 0.22.2
- numpy 1.19.2
- pandas 0.25.3
### 1- Installation

1- Create a working directory for Autogluon
```sh
$ mkdir Autogluon
$ cd Autogluon

#create a folder for the models
$ mkdir Autogluon_models

#create a folder for the Autogluon input files
$ cd ../
$ mkdir Autogluon_files
 ```
2- Use the same conda environment previously created for AutoGluon installation

```sh
$ source activate r-tensorflow
(r-tensorflow)$ python3 -m pip install -U setuptools wheel
(r-tensorflow)$ python3 -m pip install -U "mxnet<2.0.0, >=1.7.0"
(r-tensorflow)$ python3 -m pip install autogluon
  ```
For more information/problems or alternative installation methods for Autogluon installation, please visit  https://auto.gluon.ai/stable/install.html



### 2- Training

Describe the format of Autogluon files
You can run the script  ```Autogluon_models.py``` within a jupyter notebook step by step or any other python interface for the training in the AutoGluon directory.

```sh

python3 Autogluon_models.py
  ```
Describe the Autogluon jupyter notebook (mention if it includes the evaluation as well)
