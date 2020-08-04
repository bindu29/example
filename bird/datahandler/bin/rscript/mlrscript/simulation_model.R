model<-function(dataframe,variable){
  names(dataframe) <- gsub(x = names(dataframe),pattern = "\\.",replacement = "_")
  #dataframe <- data.frame(findAndTransformDates(dataframe, cols = "auto", formats = NULL,n_test = 30, ambiguities = "IGNORE", verbose = TRUE))
  pos<-names(dplyr::select_if(dataframe,is.POSIXct))
  if(length(pos)>=1){
    dats = as.data.frame(dats)
    dats<-dataframe[ , -which(names(dataframe) %in% c(pos))]
  }else{
    dats<-dataframe
  }
  names(dats) <- gsub(x = names(dats),pattern = "\\.",replacement = "_")
  dats = as.data.frame(dats)
  dats_test<-as.data.frame(dats[,-which(names(dats)==variable)])
  names(dats_test) <- gsub(x = names(dats_test),pattern = "\\.",replacement = "_")
  df.hex <- as.h2o(dats)
  df.hex_test<-as.h2o(dats_test)
  if(class(dats[,which(names(dats)==variable)])!="factor"){
    print("We are performing regression analysis")
    Actual<-dataframe[,variable]
    data = as.data.frame(dats)
    model.gaussian<-h2o.glm(y=variable, x=names(dats[,-which(names(dats)==variable)]), training_frame = df.hex, family="gaussian")
    model<-model.gaussian
    yhat <- h2o.predict(model.gaussian, df.hex_test)
    preds<-as.data.frame(yhat)
    dataframe<-cbind(dataframe,preds)
    dataframe$Actual<-Actual
    names(dataframe)[names(dataframe) == 'predict'] <- 'Predicted_values'
    dataframe = as.data.frame(dataframe)
    Differences<-dataframe[,which(names(dataframe)=="Actual")]-dataframe$Predicted_values
    dataframe$difference_Actual_predicted<-Differences
    dataframe$percentage<-round((dataframe$difference_Actual_predicted/dataframe$Actual)*100)
    Unexpected_vals<-subset(dataframe,percentage>=60|percentage<=(-60))
    Model_quality<-h2o.r2(model.gaussian)
    math<-data.frame(h2o.coef(model.gaussian))
    math<-setDT(math, keep.rownames = TRUE)[]
    coeffs<-subset(math,h2o.coef.model.gaussian.>0|h2o.coef.model.gaussian.<0)
    coeffs$rn <- gsub(x = coeffs$rn ,pattern = "\\.",replacement = "__")
    cof<-separate(data = coeffs, col = rn, into = c("left", "right"), sep = "\\__")
    cof$right[is.na(cof$right)] <- as.character(cof$left[is.na(cof$right)])
    list_pred<-list(dataframe,Unexpected_vals,Model_quality,cof)
    return(list_pred)
  }else{
    print("We are performing Classification")
    df.gbm <- h2o.gbm(y = variable, x = names(dats[,-which(names(dats)==variable)]), training_frame =df.hex, ntrees = 5,max_depth = 3,min_rows = 2, learn_rate = 0.2)
    model<-df.gbm
    yhat_classif <- h2o.predict(df.gbm, df.hex)
    preds<-as.data.frame(yhat_classif)
    predicted_vals<-colnames(preds)[apply(preds,1,which.max)]
    predicted_values <- gsub(x = predicted_vals ,pattern = "\\.",replacement = " ")
    Actual<-subset(dataframe,select=variable)
    names(Actual)[names(Actual) == variable] <- 'Actual'
    dataframe<-cbind(dataframe,predicted_values,Actual)
    Unexpected_vals<-data.frame()
    for(i in 1:nrow(dataframe)){
      if(dataframe$Actual[i]!=dataframe$predicted_values[i]){
        Unexpected_vals<-rbind(Unexpected_vals,dataframe[i,])
      }
    }
    conf_matrix<-confusionMatrix(dataframe$Actual,dataframe$predicted_values)
    Model_quality<-conf_matrix$overall[1]
    #dataframe1 <- dataframe[which(names(dataframe))==c("Actual","Predicted_values")]
    list_pred<-list(dataframe,Unexpected_vals,Model_quality)
  }
  return(list_pred)
}