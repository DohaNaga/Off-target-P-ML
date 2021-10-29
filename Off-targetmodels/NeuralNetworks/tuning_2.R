

#initiating flags
FLAGS <- flags(
  flag_integer("dense_units1",2048),
  flag_numeric("dropout1",0.1),
  flag_numeric("lr",0.01),
  flag_integer("dense_units2",1024),
  flag_numeric("dropout2",0.2),
  flag_integer("batch_size",128),
  flag_string("activation_1","relu")
)


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


tensorflow::tf$random$set_seed(42)

model <- keras_model_sequential()

model %>%
  layer_dense(units = FLAGS$dense_units1, activation = FLAGS$activation_1 ,
              input_shape = c(1024), kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$dense_units2, activation = FLAGS$activation_1 ,kernel_regularizer = regularizer_l2(l = 0.001))  %>% 
  layer_dropout(rate = FLAGS$dropout2)%>%
  layer_dense(units = 1, activation = 'sigmoid')

#compiling model

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(FLAGS$lr),
  metrics = c('binary_accuracy',balanced_acc)
)

early_stop <- callback_early_stopping(monitor = "val_balanced_acc", mode = "max", patience = 30)
callback_check_pt <- callback_model_checkpoint(paste0(targets_dataset[i],"checkpoints") ,monitor = "val_balanced_acc", verbose = 0,
                                               save_best_only = TRUE, save_weights_only = FALSE, mode = "max")



reduce_lr_on_plateau <- callback_reduce_lr_on_plateau(
  monitor = "val_balanced_acc",
  factor = 0.1,
  patience = 10,
  verbose = 0,
  mode =  "max",
  min_delta = 1e-04,
  cooldown = 0,
  min_lr = 0
)


epochs  <- 250


# Fit the model and store training stats
history <- model %>% fit(
  x_train,
  y_train,
  epochs = epochs,
  validation_split = 0.2,
  batch_size = FLAGS$batch_size,
  verbose = 1,
  callbacks = list(early_stop,callback_check_pt,reduce_lr_on_plateau)
)

history$params$epochs <- history$metrics$loss %>% length
plot(history)
#ggsave(history,paste0(targets_dataset[i],"png"))
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)

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


#saving model
save_model_tf(model,include_optimizer = TRUE, paste0(targets_dataset[i],".h5"))

