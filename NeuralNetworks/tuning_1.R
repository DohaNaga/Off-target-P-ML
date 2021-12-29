##############loading libraries
library(reticulate)
library(tensorflow)
library(keras)
library(tfruns)
library(dplyr)
library(caret)
library(xlsx)
library(readxl)
library(readr)
#name of previously created conda  environment
use_condaenv("r-tensorflow")

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



#reading necessary datasets for models
activities_dataset = read_excel("../Datasets/dataset1_curated.xlsx")
targets_dataset = activities_dataset$OFF_TARGET %>% unique()
fingerprints_dataset = data.matrix(read.table("../Datasets/dataset_2.csv") )


for (i in 1 : length(targets_dataset)) {
  
  #last number in bracket is included
  set.seed(42)
  tensorflow::tf$random$set_seed(42)
  
  target_name <- targets_dataset[i]
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
  
  
  #labels binary fps
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
  
  tensorflow::tf$random$set_seed(42)
  
  #tuning
  tuning_run('tuning_2.R',
             runs_dir = paste('tuning/',targets_dataset[i]),
             confirm = FALSE   ,sample = 0.5, flags =  list(
               dropout1 = c(0,0.1,0.2),
               dense_units1 = c(256,512,1024,2048),
               dropout2 = c(0.2,0.3,0.4),
               batch_size = c(64,128,256),
               dense_units2 = c(256,512,1024,2048),
               lr = c(0.01,0.001,0.0001)
             ))
  
  #copying runs with best balanced accuracy
  copy_run( ls_runs(runs_dir = paste('tuning/', targets_dataset[i]),
                    order = NA..1 ,decreasing = TRUE)[1:1,], to = paste('tuning/', targets_dataset[i],'best_runs_ba'))
  
  
  #copying runs with best accuracy
  copy_run( ls_runs(runs_dir = paste('tuning/', targets_dataset[i]),
                    order = NA. ,decreasing = TRUE)[1:1,], to = paste('tuning/', targets_dataset[i],'best_runs_acc'))
  
  
  #copying runs with best loss
  copy_run( ls_runs(runs_dir = paste('tuning/', targets_dataset[i]),
                    order = eval_ , decreasing = FALSE)[1:1,], to = paste('tuning/', targets_dataset[i],'best_runs_loss'))
  
  
  info_all_runs <-  ls_runs(runs_dir = paste('tuning/', targets_dataset[i]),
                            order = NA..1, decreasing = TRUE)
  
  #creating a folder for the grid info
  
  dir.create("tuning/grid_inforuns/")
 
   #writing infromation on all runs for each target in a file called "targetname_grid_inforuns.xls"
  write.xlsx(info_all_runs , paste("tuning/grid_inforuns/",targets_dataset[i], ".xlsx"))
  
  
  #cleaning rest of runs
  clean_runs(runs_dir = paste('tuning/'
                              , targets_dataset[i]), confirm = FALSE )
  
  #removing permanently resy of runs
  purge_runs(runs_dir = paste('tuning/', targets_dataset[i]),confirm = FALSE)
  
  
}  


