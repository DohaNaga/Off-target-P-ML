library(tensorflow)
library(keras)
library(tfruns)
library(caret)
library(dplyr)
library(ggpubr)
library(yardstick)
library(PRROC)
library(readxl)
library(mccr)
library(xlsx)

#reading necessary datasets for models
activities_dataset = read_excel("../Datasets/dataset1_curated.xlsx")
targets_dataset = activities_dataset$OFF_TARGET %>% unique()
fingerprints_dataset = data.matrix(read.table("../Datasets/dataset_2.csv"))

#creating directories for plots

dir.create("tuning/plots/")
dir.create("tuning/plots/AUC")
dir.create("tuning/plots/AUCPR")

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


nn_bestba_allmetrics = NULL
for (i in 1 : length(targets_dataset)) {
  set.seed(42)
  tensorflow::tf$random$set_seed(42)
   
  Target <- targets_dataset[i]
  #importing from original dataset
  input_target_df <- (activities_dataset %>% dplyr::select(COMPOUND_ID,OFF_TARGET,BINARY_VALUE))%>%
    filter(OFF_TARGET == targets_dataset[i])
  
  
  #########stratified splitting
  train.index <- createDataPartition(input_target_df$BINARY_VALUE, p = .8, list = FALSE)
  #creating balanced splits
  #activities
  y_train_df <- input_target_df[train.index,]
  
  #reshape into a matrix  
  #removing the COMPOUND_ID  coloumn & target coloumn
  y_train_mt <- y_train_df[,3]
  
  y_train_mt <- data.matrix(y_train_mt)
  
  
  #putting back rownames
  rownames(y_train_mt) <- y_train_df$COMPOUND_ID 
  y_train <- y_train_mt
  
  
  #labels binary 
  x_train <-fingerprints_dataset[y_train_df$COMPOUND_ID ,]
  
  #test sets
  y_test_df  <-   input_target_df[-train.index,]
  #reshape into a matrix  
  #removing the COMPOUND_ID  coloumn & target coloumn
  y_test_mt <- y_test_df[,3]
  
  y_test_mt <- data.matrix(y_test_mt)
  #putting back rownames
  rownames(y_test_mt) <- y_test_df$COMPOUND_ID 
  y_test <- y_test_mt
  
  x_test <-fingerprints_dataset[y_test_df$COMPOUND_ID ,]
  #loading saved models from the tuning folder
 
  run_dir_name <- (ls_runs(runs_dir = paste("tuning/",targets_dataset[i],"best_runs_ba"))) $run_dir
  model_path <-  paste(run_dir_name,"/",targets_dataset[i],".h5",sep = "") 
  
  model <- load_model_tf(model_path, custom_objects = list("balanced_acc"= balanced_acc))
  
  
  ####4 testing model
  eval_model <- model %>% evaluate(x_test , y_test )
  
  pred_model <- model %>% predict_classes(x_test)
  
  rownames(pred_model) <- rownames(x_test)
  
  #calculate probabilities for curves
  proba_model <- model %>% predict_proba(x_test)
  
  
  #calculation of confusion matrix
  #checking evaluation metrics
  #define positive as 1 in confusion matrix
  
  conf.matrix <-  caret::confusionMatrix(
    factor(pred_model, levels = 0:1),
    factor(y_test, levels = 0:1),
    positive = "1"
  )
  
  BalancedAccuracy <- conf.matrix$byClass[11][[1]]
  Accuracy <- conf.matrix$overall[1][[1]]
  
  # can add f1 score if needed also
  #F1 <- conf.matrix$byClass[7]
  #mathews correlation coefficent
  MCC <- mccr(y_test,pred_model)
  
  
  # auc and pr curves
  
  probs = as.vector(proba_model)
  truth = y_test %>% data.frame()
  fg <- probs[truth$BINARY_VALUE == 1]
  bg <- probs[truth$BINARY_VALUE == 0]
  
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  AUC <- roc[2]$auc

   #do not plot auc if its nan, will cause error and halt execution
  if(is.na(AUC) == TRUE){

   AUC <- "NaN"
   
  }else{
    
   # ROC Curve
    png(filename = paste0("tuning/plots/AUC/","AUC_PLOT_",targets_dataset[i],".png"))
    plot(roc)
    dev.off()
  
  }  
    

    # PR Curve

  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  AUCPR <- pr[2]$auc.integral
  
  #do not plot aucpr if nan, will cause error and halt execution
   if(is.na(AUCPR) == TRUE){
    
    
    AUCPR <- "NaN"
    
    }else{
     
     # saving aucpr plots
      png(filename = paste0("tuning/plots/AUCPR/","AUCPR_PLOT_",targets_dataset[i],".png"))
      plot(pr)
      dev.off()
 
    }


    nn_bestba_allmetrics = rbind(nn_bestba_allmetrics, data.frame(Target,BalancedAccuracy,Accuracy,MCC, AUC, AUCPR))
}

write.xlsx(nn_bestba_allmetrics,"tuning/neuralnetworks_allmetrics.xlsx")



