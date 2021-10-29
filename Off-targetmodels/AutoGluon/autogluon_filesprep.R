##datasets

activities_dataset = read.xls("../Datasets/dataset_1.xls")
targets_dataset = dataset_1$OFF_TARGET %>% unique()
fingerprints_dataset = read.xls("../Datasets/dataset_2.xls")


for (i in 1: length(targets_dataset)){
  #importing from original dataset
  
  
  input_target_df <- (activities_dataset %>% select(CAS.Number,OFF_TARGET, BINARY_VALUE))%>%
  filter(OFF_TARGET == targets_dataset [i])
  
  CAS.Numbers.targ <- input_target_df$CAS.Number
  
  fps_input_comps <-   fingerprints_dataset[CAS.Numbers.targ,] %>% data.frame()
  fps_input_comps$CAS.Number <- rownames(fps_input_comps)
  
autogluon_binary<-  full_join(fps_input_comps,input_target_df) 
  
  #renaming into id
autogluon_binary<- dplyr::rename(autogluon_binary , "ID" = "CAS.Number")
  train.index <- createDataPartition(autogluon_binary$BINARY_VALUE, p = .8, list = FALSE)
  #reproducible
  set.seed(42)
  
  #creating balanced splits
  autogluon_binary_train <- autogluon_binary[ train.index,]
  
  autogluon_binary_test <- autogluon_binary[-train.index,]
  
  #writing into csv
  write.csv( x = autogluon_binary_test,file = paste0("/Autogluon_files/" , "test_",gsub(" ","",targets_dataset[i]),".csv") ,row.names = FALSE)
  
  write.csv( x = autogluon_binary_train,file = paste0("/Autogluon_files/" , "train_",gsub(" ","",targets_dataset[i]),".csv") ,row.names = FALSE)
  
}

