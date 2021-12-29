#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

#name of previously created conda  environment
library(reticulate)
use_condaenv("r-tensorflow")
library(tensorflow)
library(keras)
library(caret)
library(dplyr)
library(rcdk)
library(fingerprint)
library(readxl)
library(mccr)
library(xlsx)
library(magicfor)
magic_for(print,progress = TRUE)


###custom functions
balanced_acc <- custom_metric("balanced_acc",function(y_true,y_pred){
  y_pred_pos = k_round(k_clip(y_pred, 0, 1))
  y_pred_neg = 1 - y_pred_pos
  
  y_pos = k_round(k_clip(y_true, 0, 1))
  y_neg = 1 - y_pos
  
  tp = k_sum(y_pos * y_pred_pos)
  tn = k_sum(y_neg * y_pred_neg)
  
  fp = k_sum(y_neg * y_pred_pos)
  fn = k_sum(y_pos * y_pred_neg)
  
  sensi = (tp/(tp + fn + k_epsilon()))
  specifi = (tn/(tn + fp + k_epsilon()))
  
  return((sensi + specifi )/ 2 )
  
})

#test set
external_test <- read.csv(args[1], stringsAsFactors = FALSE)         

#targets 
target_names = (read.csv("target_names.csv") ) %>% pull(Target)

#calculate mols

external_test_mols <- sapply(external_test$SMILES,parse.smiles)

###Calculate ecfp4

external_test_ecfp4 <- lapply(external_test_mols[1:length(external_test_mols)], get.fingerprint,type = 'circular', circular.type = 'ECFP4') 

#add srns  & convert fps objects into  matrix 
external_test_ecfp4_binary <- fp.to.matrix(external_test_ecfp4)
rownames(external_test_ecfp4_binary) <- external_test$COMPOUND_ID

magic_for(print,progress = TRUE)

for (i in 1 : 47) {
  set.seed(42)
  tensorflow::tf$random$set_seed(42)
  
  model_path <-  paste(target_names[i],".h5",sep = "") 
  model <- load_model_tf(model_path, custom_objects = list("balanced_acc"= balanced_acc))
  

  #predit for new compounds
  pred_model <- model %>% predict_classes(external_test_ecfp4_binary) 
  rownames(pred_model) <- rownames(external_test_ecfp4_binary) 
  colnames(pred_model) <- target_names[i]
  print(pred_model)
  
}

predictions_ba <- magic_result()

predictions <- predictions_ba$pred_model %>% data.frame(check.names = FALSE) 

#write the predictions
write.xlsx(predictions, "predictions.xls",row.names = TRUE)


