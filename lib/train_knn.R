train <- function(feature_df = pairwise_data, par = NULL){
  k = par$k


model <- knn(train = dat_train[,-which(names(dat_train) == 'emotion_idx')], 
             test = dat_test[,-which(names(dat_test) == 'emotion_idx')], 
             cl = dat_train$emotion_idx, 
             k = K)

return(model)
}