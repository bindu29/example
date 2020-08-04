#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param yValue
#* @param parametersObj
#* @get /linear_regression
linear_regression<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",yValue="",parametersObj="") {
  lapply(list('jsonlite','devtools','ps','Rtools','DBI','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  features<-c(names(data));
  correlationData<-correlations(data,features);
  complete_data<-data.frame(correlationData[3])
  y=c(yValue)
  new<-transformation(complete_data,y)
  vars<-variable_importance_h2o(data,y)
  if(class(vars)=="data.frame"){
  important_vars<-vars$x
  subset_data <- new[,which(colnames(new)%in%important_vars)]
  subset_data[,y]<-new[,y]
  
  if(parametersObj!="NONE"){
    params_obj<-fromJSON(parametersObj);
    balance_classes_value = FALSE;
    if(params_obj$balance_classes=="TRUE"){
      balance_classes_value=TRUE;
    }
    print("with user defined parameters")
    parameters=c(nfolds=params_obj$nfolds,family=params_obj$family,balance_classes=balance_classes_value)
  }else{
    print("defaul parameters")
    parameters=c(nfolds=5,family="gaussian",balance_classes=FALSE)
  }
  dfs<-modelling(subset_data,y,parameters,vars)
  }else{
   subset_data<-new
   if(parametersObj!="NONE"){
     params_obj<-fromJSON(parametersObj);
     balance_classes_value = FALSE;
     if(params_obj$balance_classes=="TRUE"){
       balance_classes_value=TRUE;
     }
     print("with user defined parameters")
     parameters=c(nfolds=params_obj$nfolds,family=params_obj$family,balance_classes=balance_classes_value)
   }else{
     print("defaul parameters")
     parameters=c(nfolds=5,family="gaussian",balance_classes=FALSE)
   }
   dfs<-modelling(subset_data,y,parameters,vars)
  }
  dbDisconnect(con);
  remove(con);
  dfs
}