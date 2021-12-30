library(dplyr)
library(readxl)
library(caret)
library(xlsx)
#creating a folder for the Autogluon files
dir.create("Autogluon_files/")
#reading necessary datasets for models
activities_dataset = read_excel("../Datasets/dataset_1.xlsx")
targets_dataset = activities_dataset$OFF_TARGET %>% unique()
fingerprints_dataset = data.matrix(read.table("../Datasets/dataset_2.csv") )



for (i in 1: length(targets_dataset)){
  #importing from original dataset
  
  
  input_target_df <- (activities_dataset %>% select(CAS.Number,OFF_TARGET, BINARY_VALUE))%>%
    filter(OFF_TARGET == targets_dataset [i])
  
  CAS.Numbers.targ <- input_target_df$CAS.Number
  
  fps_input_comps <-   fingerprints_dataset[CAS.Numbers.targ,] %>% data.frame()
  fps_input_comps$CAS.Number <- rownames(fps_input_comps)
  
  autogluon_binary<-  full_join(fps_input_comps,input_target_df) 
  
  #renaming into id as required by AutoGluon
  autogluon_binary<- dplyr::rename(autogluon_binary , "ID" = "CAS.Number")
  
  #reproducible
  set.seed(42)
  train.index <- createDataPartition(autogluon_binary$BINARY_VALUE, p = .8, list = FALSE)
  
  
  #creating balanced splits
  autogluon_binary_train <- autogluon_binary[ train.index,]
  
  autogluon_binary_test <- autogluon_binary[-train.index,]
  
  #writing into csv
  write.csv( x = autogluon_binary_test,file = paste0("Autogluon_files/" , "test_",gsub(" ","",targets_dataset[i]),".csv") ,row.names = FALSE)
  
  write.csv( x = autogluon_binary_train,file = paste0("Autogluon_files/" , "train_",gsub(" ","",targets_dataset[i]),".csv") ,row.names = FALSE)
  
}

