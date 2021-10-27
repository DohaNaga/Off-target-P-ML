
# Off-target-modelling
This repository contains the necessary scripts to derive off-target models using (1) A neural network framework (2)An autmomated machine learning framework (via Autogluon).




## 1- Preparation of the models input

ECFP4 fingerprints are used for the predictions of the binary activities of the structures. These fingerprints need to be created as a first step and will be used as an input for the training of both the neural networks and the autogluon models.

Use the script `fingerprints_preparation.R` to generate the ECFP4 fingerprints for the required dataset (in this case `dataset_1`)

The script is tested under R version 3.5.1 in R studio version 1.1.456.
##### Dependencies : 
- R 3.5.1
- rcdk 3.5.0
- rcdklibs 2.3


## 2- Neural networks models
The script is tested under R version 3.5.1 in R studio version 1.1.456.

##### Dependencies : 
- Python ≥ 3.6
- reticulate 1.16
- Tensorflow 2.2.0
- Keras 2.3.0
- Tfruns 1.4


### Installation


1- Create a working directory
  ```sh
mkdir tuning
  ```

2. Create a conda working environment from the unix command line

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

For more information/problems regarding Tensorflow installation in R or alternative installation methods, please visit https://tensorflow.rstudio.com/installation/

### Training

Describe that you can use it for this specific target list or for another one

Describe the training files (two files) , describe the output files generated , the sh file 



### Evaluation 
Describe the evaluation file



## 2- AutoGluon models
The script is tested under Python version 3.6.5 in Jupyter notebook  version 7.12.0. 

##### Dependencies : 
- Python ≥ 3.6
- MXNet ≥ 1.7.0.
- Autogluon 0.0.13
- sklearn 0.22.2
- numpy 1.19.2
- pandas 0.25.3
### Installation

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



### Training

Describe the format of Autogluon files
You can run the script  ```Autogluon_models.py``` within a jupyter notebook step by step or any other python interface for the training in the AUTOGLUON directory.

```sh

python3 Autogluon_models.py
  ```
Describe the Autogluon jupyter notebook (mention if it includes the evaluation as well)
