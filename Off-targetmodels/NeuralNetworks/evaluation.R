library(tensorflow)
library(keras)
library(magicfor)
library(tfruns)
library(caret)
library(dplyr)
library(ggpubr)
library(yardstick)
library(PRROC)
magic_for(print,progress = TRUE)

#reading necessary datasets for models
activities_dataset = read.xls("../Datasets/dataset_1.xls")
targets_dataset = dataset_1$OFF_TARGET %>% unique()
fingerprints_dataset = read.xls("../Datasets/dataset_2.xls")

for (i in 1 : 50) {
  set.seed(42)
  tensorflow::tf$random$set_seed(42)
  
  Target <- targets_dataset[i]
  #importing from original dataset
  input_target_df <- (activities_dataset %>% dplyr::select(CAS.Number,OFF_TARGET,BINARY_VALUE))%>%
    filter(OFF_TARGET == targets_dataset[i])
  
  
  #########stratified splitting
  train.index <- createDataPartition(input_target_df$BINARY_VALUE, p = .8, list = FALSE)
  
  #creating balanced splits
  #activities
  y_train_df <- input_target_df[train.index,]
  
  #reshape into a matrix  
  #removing the CAS.Number  coloumn & target coloumn
  y_train_mt <- y_train_df[,3]
  
  y_train_mt <- data.matrix(y_train_mt)
  
  
  #putting back rownames
  rownames(y_train_mt) <- y_train_df$CAS.Number 
  y_train <- y_train_mt
  
  
  #labels binary fps
  x_train <- fps_ecfp4_binary[y_train_df$CAS.Number ,]
  
  #test sets
  y_test_df  <-   input_target_df[-train.index,]
  #reshape into a matrix  
  #removing the CAS.Number  coloumn & target coloumn
  y_test_mt <- y_test_df[,3]
  
  y_test_mt <- data.matrix(y_test_mt)
  #putting back rownames
  rownames(y_test_mt) <- y_test_df$CAS.Number 
  y_test <- y_test_mt
  
  x_test <- fps_ecfp4_binary[y_test_df$CAS.Number ,]
  
  
  #loading saved models from the tuning folder
 
  run_dir_name <- (ls_runs(runs_dir = paste("/tuning/",targets_dataset[i],"best_runs_ba"))) $run_dir
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
  
  BalancedAccuracy <- conf.matrix$byClass[11]
  Accuracy <- conf.matrix$overall[1]
  F1 <- conf.matrix$byClass[7]
  #mathews correlation coefficent
  MCC <- mccr(y_test,pred_model)
  
  
  
  # #correct way to calculate area under  auc & pr curves
  #   
  probs = as.vector(proba_model) 
  truth = y_test %>% data.frame()
  fg <- probs[truth$. == 1]
  bg <- probs[truth$. == 0]
  # 
  #   

#creating directories for plots

dir.create("tuning/plots/")
dir.create("tuning/plots/AUC")
dir.create("tuning/plots/AUCPR")

  #  # ROC Curve    
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  AUC <- roc[2]$auc
  plot(roc)
  #  
  ggsave(paste0("AUC_PLOT",targets_dataset[i],".png"), roc,"tuning/plots/AUC/"
                  ,device = NULL, dpi =800)

  #  
  #  # PR Curve
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  AUCPR <- pr[2]$auc.integral
  plot(pr)
  
  #saving aucpr plots  
 ggsave(paste0("AUCPR_PLOT",targets_dataset[i],".png"), roc,"tuning/plots/AUCPR/"
         ,device = NULL, dpi =800)
  
  print(Target)
  print(BalancedAccuracy)
  print(Accuracy)
  print(AUC)
  print(AUCPR)
  print(MCC)
  
  
}


#dataframe with all metrics for neural networks method
nn_bestba_allmetrics <- magic_result_as_dataframe()

write.xls(nn_bestba_allmetrics,"neuralnetworks_allmetrics.xls")



