modelling<-function(df,y,parameters="None",vars,features,final_simulation,model_coff,test_data,columnsArray){
  fin_names<-names(df)
  features<-features[features!=y]
  dependent_y<-y
  df<-as.data.frame(subset(df,select=c(features,dependent_y)))
  df1<-df
  x<-names(df1)
  x1 <-  gsub('([[:punct:]])|\\s+','',names(df1))
  names(df1)<-x1
  if(length(df1)>2){
    x1 <-  gsub('([[:punct:]])|\\s+','',names(df1))
    y_vif<- gsub('([[:punct:]])|\\s+','',y)
    vars_vif<-vif_func(df1[,-which(names(df1)==y_vif)],thresh=6,trace=T)
    df1<-as.data.frame(df1[,vars_vif[[1]]])
    names(df1)<-vars_vif[[1]]
    rem_cols<-unlist(vars_vif[[2]])
    dif_col <- setdiff(names(df1),rem_cols)
    if(!is.null(dif_col)){
      lis_cols<-list()
      for(i in 1:length(x1)){
        for(j in 1:length(dif_col)){
          if(x1[i] == dif_col[j]){
            lis_cols[[i]] <- i
          }
        }
      }
      fin_index<-c(unlist(lis_cols))
      df1[,y]<-df[,y]
      left_names<-names(df[,as.numeric(fin_index)])
      left_names<-append(left_names,y)
      fin_names<-left_names
    }else{
      df1[,y]<-df[,y]
    }
  }else{
    df1<-df1
  }
  df<-df1
  features<-names(df)
  set.seed(101)
  sample <- sample.int(n = nrow(df), size = floor(.75*nrow(df)), replace = F)
  train <- df[sample,]
  test  <- df[-sample,]
  test_x<-subset(test,select=c(features))
  if(ncol(train)>1){
    df.hex <- as.h2o(train)
    #df.hex
    fea1<-features[features!=y]
    dats_test<-data.frame(test[,-which(names(test)==y)])
    names(dats_test)<-fea1
    df.hex_test <- as.h2o(dats_test)
    Actual <- test %>% select(dependent_y)
    if(parameters=="None"){
      print("defaul parameters")
      df_x <- select(df, -dependent_y)
      model.fit<-h2o.glm(y=y, x=names(df_x), training_frame = df.hex,nfolds=5,family="gaussian",balance_classes=FALSE)
    }else{
      print("with user defined parameters in h20")
      params_obj<-fromJSON(parameters);
      balance_classes_value = FALSE;
      if(tolower(params_obj$balance_classes)=="true"){
        balance_classes_value=TRUE;
      }
      print("nfolds")
      print(params_obj$nfolds)
      print("family")
      print(params_obj$family)
      print("balance_classes")
      print(balance_classes_value)
      model.fit<-h2o.glm(y=y, x=names(as.data.frame(df[,-which(names(df)==y)])), training_frame = df.hex,nfolds=params_obj$nfolds,family=params_obj$family,balance_classes=balance_classes_value)
    }
    RMSE<-h2o.rmse(model.fit)
    R_squared<-h2o.r2(model.fit)
    model<-model.fit
    yhat <- h2o.predict(model.fit, df.hex_test)
  }else{
    form=as.formula(paste(y,"~."))
    model.fit<-lm(form, data = df)
    yhat<-predict(model,test_x)
    RMSE<-sqrt(mean(model$residuals^2))
    Residuals<-model$residuals
    R_squared<-summary(model.fit)$r.squared
  }
  R_squared_rounded<-round(R_squared*10)
  preds<-as.data.frame(yhat)
  dataframe<-test
  names(dataframe)<-fin_names
  dataframe<-cbind(dataframe,preds)
  dependent_y<-y
  dataframe$Actual<-Actual[,which(names(Actual)==dependent_y)]
  names(dataframe)[names(dataframe) == 'predict'] <- 'Predicted_values'
  Differences<-dataframe[,which(names(dataframe)=="Actual")]-dataframe$Predicted_values
  dataframe$difference_Actual_predicted<-Differences
  dataframe$percentage<-round((dataframe$difference_Actual_predicted/dataframe$Actual)*100)
  Unexpected_vals<-subset(dataframe,percentage>=60|percentage<=(-60))
  #mape_value<-mean(abs((dataframe$Actual-dataframe$Predicted_values)/dataframe$Actual) * 100)
  Actual_predicted<-data.frame(dataframe$Predicted_values, dataframe$Actual)
  name<-names(dataframe)
  name<-name[name!=c("Actual")]
  name<-name[name!=c("Predicted_values")]
  name<-name[name!=c("difference_Actual_predicted")]
  dataframe<-dataframe[,c("Actual","Predicted_values","difference_Actual_predicted",name)]
  dataframeNewSet<-dataframe
  #renames columns of a dataframe
  for(i in 1:length(names(dataframeNewSet))){
    for(j in 1:length(row.names(columnsArray))){
      if(names(dataframeNewSet)[i] == (columnsArray["changedColumnName"][j,1])){
        names(dataframeNewSet)[i] <- columnsArray["columnDisplayName"][j,1]
      }
    }
  }
  print(dataframeNewSet)
  listed<-list(R_squared_rounded,dataframeNewSet,Actual_predicted,vars,RMSE,R_squared,final_simulation,model_coff,test_data)
  return(listed)
}