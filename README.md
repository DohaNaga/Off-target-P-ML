
# Off-target-modelling
This repository contains the necessary scripts to derive off-target models using (1) A neural network framework (2)An autmomated machine learning framework (via Autogluon).




## Preparation of the models input

- ECFP4 fingerprints are used for the predictions of the binary activities of the structures.
- Use the script `fingerprints_preparation.R` to generate the ECFP4 fingerprints for the required structures (in this case `dataset_1`)


## 1- Neural networks models

### Create a working directory
  ```sh
mkdir tuning
  ```



### Installation

1. Create a conda working environment from the unix command line

  ```sh

#creat conda environment using python 3.6

$ conda create -n r-tensorflow pip python = 3.6

#activate  environment

$ source activate r-tensorflow


  ```
2. once env activated, install keras, tensorflow and tfruns from the environment

  ```sh
(r-tensorflow)$ pip install tensorflow

(r-tensorflow)$ pip install keras

(r-tensorflow)$ pip install tfruns
  ```

3. Testing if Tensoflow and Keras are successfuly installed

#Opening R version 3.5.1 from the terminal 


  ```sh
$  ml R/3.5.1-goolf-1.7.20 

  ```

4. loading keras and tensorflow libraries in R

  ```{r}  
use_condaenv("r-tensorflow")
library(keras)
library(tensorflow)
library(tfruns)

```

5. Testing if tensorflow is working in R 

```{r}  

mnist <- dataset_mnist()

x_train <- mnist$train$x

head(x_train)

#should get back a matrix means th
  [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13] [,14] [,15] [,16] [,17] [,18] [,19] [,20] [,21] [,22] [,23]

```









Describe that you can use it for this specific target list or for another one

Describe the training files (two files) , describe the output files generated , the sh file 

Describe the evaluation file



## 2- AutoGluon models

Describe Autogluon installation

Describe the format of Autogluon files

Describe the Autogluon jupyter notebook (mention if it includes the evaluation as well)
