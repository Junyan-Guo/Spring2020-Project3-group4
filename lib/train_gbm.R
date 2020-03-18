gbm_train<-function(dat_train){
  
  library(gbm)
  
  tm.train<-system.time(gbm.fit<-gbm(emotion_idx~.,
                                     distribution="multinomial",
                                     data=dat_train,
                                     n.trees = 200,
                                     bag.fraction=0.65,
                                     shrinkage = 0.1,
                                     cv.folds=3))
  
  return(list(gbm.fit,tm.train))
}