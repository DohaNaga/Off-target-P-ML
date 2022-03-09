library(dplyr)
library(caret)
library(randomForest)
library(PRROC)
library(mccr)
library(magicfor)
library(utils)

###custom function for the traincontrol of the random forest
###adapt the "Twoclassfunction"  to use balanced accuracy or mcc as a tuning metric

custom_rf <-   function (data, lev = NULL, model = NULL) 
{
  sensi <- sensitivity(data[, "pred"], data[, "obs"], 
                       lev[1])
  specifi <- specificity(data[, "pred"], data[, "obs"], lev[2])
  
  Balancedacc <- (sensi+specifi)/2
  mcc <- mccr(data[, "pred"], data[,"obs"])
  out <- c(Balancedacc,mcc)
  names(out) <- c("Balancedaccuracy","mcc")
  out
}


#reading necessary datasets for models
activities_dataset = read_excel("../Datasets/dataset1_curated.xlsx")
targets_dataset = activities_dataset$OFF_TARGET %>% unique()
fingerprints_dataset = data.matrix(read.table("../Datasets/dataset_2.csv") )


activities_dataset = binding_pctinhb_human_recent
fingerprints_dataset =  fps.cerep.recent.binary

magic_for(print,progress = TRUE)
for (i in 1 : 50) {
  
  target_name <- targets_dataset[i]
  #importing from original dataset
  input_target_rf <- (activities_dataset %>% dplyr::select(COMPOUND_ID,OFF_TARGET,BINARY_VALUE))%>%
    filter(OFF_TARGET == targets_dataset[i])
  
  
  
  #fingerprints
  COMPOUND_IDS <-  rownames(fingerprints_dataset)
  fingerprints_df <- fingerprints_dataset %>% data.frame()
  fingerprints_df$COMPOUND_ID <- COMPOUND_IDS

  
   #merging off-target binary outcomes to be predicted with the fingerprints dataframe
  
  input_target_rf <- inner_join(input_target_df,fingerprints_df, by = "COMPOUND_ID")
  
  input_target_rf$BINARY_VALUE <- input_target_rf$BINARY_VALUE %>% as.factor()
  
  

  
  #reproducible
  set.seed(42)
 
  #########stratified activity splitting
  train.index <- createDataPartition(input_target_rf$BINARY_VALUE, p = .8, list = FALSE)
  
  train_set <- input_target_rf[train.index,] 
    
  test_set <- input_target_rf[-train.index,]

  ##########random forest model

   rf_model <- train(BINARY_VALUE ~ . , # fingerprints as predictors
                  data = train_set[,-c(1,2)], # Use the train data frame as the training data
                  method = 'rf',
                  ntree = 100, #Fix number of trees to 100
                  metric = "Balancedaccuracy", # tuning metric (can choose BA or MCC)
                  trControl = trainControl(method = 'cv', #use cross validation
                                           summaryFunction = custom_rf, #custom function containing BA and MCC
                                           #classProbs = TRUE, #if uncomment then change binary classes to active/inactive instead of 0,1
                                             number = 10 ))   
 
    #saving the model by target name
    saveRDS(rf_model , paste("model_rf_",target_name,".RDS",sep = ""))
  
  
  #prediction on test set
  rf_pred <- predict(rf_model , test_set[,-c(1,2)])
  rf_pred_proba <- predict(rf_model , test_set[,-c(1,2)],type = "prob")
  
  rf_confmatrix  <-  caret::confusionMatrix(
    factor(rf_pred, levels = 0:1),
    factor(test_set$BINARY_VALUE, levels = 0:1),
    positive = "1"
  )

  #calculation of all performance metrics
  Acc <- rf_confmatrix$overall[[1]]
  BA <- rf_confmatrix$byClass[[11]]
  F1 <- rf_confmatrix$byClass[[7]]
  mcc <- mccr (rf_pred , test_set$BINARY_VALUE)
  #auc and aucpr 
  probs = rf_pred_proba #need to put all in one coloumn
  truth = test_set$BINARY_VALUE %>% data.frame()
  #fg <- probs[truth$. == "active",] $active
  fg <- probs[truth$. == "1",2] 
  
  #bg <- probs[truth$. == "Inactive",] $Inactive
  bg <- probs[truth$. == "0",2] 
  
  
  #   
  #  # ROC Curve    
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  AUC <- roc[2]$auc
  
  
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
  AUCPR <- pr[2]$auc.integral
  
  
  print(target_name)
  print(BA)
  print(Acc)
  print(F1)
  print(AUC)
  print(AUCPR)
  print(mcc)
  
  
}  
  
randomforest_allmetrics <- magic_result_as_dataframe()  
  
write.csv(randomforest_allmetrics,"randomforest_evaluation_metrics.csv")

