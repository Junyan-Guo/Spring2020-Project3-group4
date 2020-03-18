cv.function.gbm <- function(dat_train,shrink){
  gbm_model <- gbm(formula = emotion_idx ~.,
                   distribution = "multinomial",
                   data = dat_train,
                   cv.folds = 3,
                   shrinkage = shrink,
                   n.trees = 200)
  
  error <- c(mean(gbm_model$train.error),sd(gbm_model$train.error)) 
  
  print(error)
  return(error)
}