#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param columnsArray
#* @get /datasnapshot
datasnapshot<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",columnsArray="") {
  options(warn=-1)  
  lapply(list('jsonlite','devtools','ps','DBI','aCRM','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  features<-c(names(data));
  columnsArray <- fromJSON(columnsArray)
  changed_data <- columns_data_type(data,columnsArray)
  correlationData<-correlations(changed_data)
  if(data.frame(correlationData[5]) == "no categorical data"){
    chisquare_Data<-"no categorical data"
  }else{
    chisquare_Data<-ch_sq_test(as.data.frame(correlationData[5],optional = TRUE))
  }
  if(length(data.frame(correlationData[2]))>0){
    cor_Data<-correlationData[2];
  }else{
    cor_Data<-"no numerical data";
  }
  print(paste0("missing_columns------",correlationData[7]))
  result<-list(cor_Data,chisquare_Data,correlationData[7])
  dbDisconnect(con);
  remove(con);
  result
}

#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param yValue
#* @param xValues
#* @param parametersObj
#* @param columnsArray
#* @get /linear_regression
linear_regression<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",yValue="",xValues="",parametersObj="",columnsArray="") {
  options(warn=-1)
  lapply(list('fmsb','jsonlite','devtools','ps','aCRM','DBI','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx','MLmetrics'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  h2o.init(nthreads=-1,enable_assertions = FALSE) 
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  features<-fromJSON(xValues);
  y=c(fromJSON(yValue))
  #features <- gsub(x = features,pattern = "\\ ",replacement = "_")
  #y<-gsub(x=y,pattern = "\\ ",replacement = "_")
  columnsArray <- fromJSON(columnsArray)
  changed_data <- columns_data_type(data,columnsArray)
  correlationData<-correlations(changed_data)
  complete_data<-correlationData[3]
  new_complete_data<-complete_data[[1]]
  features<-c(names(new_complete_data));
  new<-transformation(new_complete_data,features)
  vars<-variable_importance_h2o(new,y)
  #an.error.occured <- FALSE
  # tryCatch( { vars<-variable_importance_h2o(new,y);}
  #           , error = function(e) {an.error.msg<<-"Variable importance cannot be computed with single independent variable"
  #           an.error.occured = TRUE
  #           })
  # if(an.error.occured){
  #   dbDisconnect(con);
  #   remove(con);
  #   an.error.msg
  # }
  if(class(vars)=="data.frame"){
    important_vars<-vars$x
    subset_data <- new[,which(colnames(new)%in%important_vars)]
    subset_data[,y]<-new[,y]
    tarname<-colnames(subset_data)[which(names(subset_data) == "target_var")] <- y 
  }else{
    subset_data<-new
  }
  # parametersObj<-'{"validation_frame":"TRUE","nfolds":5,"family":"gaussian","balance_classes":"FALSE"}'
  parameters ="NONE"  
  if(parametersObj!="NONE"){
    parameters = parametersObj;
  }
  features<-names(subset_data)
  model_data<-model(subset_data,tarname)
  subset_data1<-subset_data[,-which(names(subset_data)==y)]
  simulation_var<-data_str(subset_data1)
  model_coff<-data.frame(model_data[4])
  test<-subset_data[sample(nrow(subset_data), 1), ]
  test<-test[,-which(names(test)==y)]
  final_simulation<-simulation(test,model_coff)
  dfs<-modelling(subset_data,y,parameters,vars,features = features,final_simulation,model_coff,test,columnsArray)
  dbDisconnect(con);
  remove(con);
  dfs
}


#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param parametersObj
#* @param columnsArray
#* @get /clustering
clustering<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",parametersObj="",columnsArray=""){
  lapply(list('jsonlite','fpc','aCRM','dummies','jsonlite','devtools','ps','DBI','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx','MLmetrics'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  h2o.init()
  print(parametersObj)
  columnsArray <- fromJSON(columnsArray)
  changed_data <- columns_data_type(data,columnsArray)
  correlationData<-correlations(changed_data)
  complete_data<-as.data.frame(correlationData[3])
  features<-names(complete_data)
  if(length(complete_data)>0){
    new<-transformation(complete_data,features)
    print("======= done transformation =======")
    parameters="None"
    if(tolower(fromJSON(parametersObj))!="none"){
      parameters = parametersObj;
    }
    dfs<-kmeansmodel(new,parameters)
    print("======== Done clustering ========")
    cluster_profile_data<-cluster_profiling(data.frame(dfs[1]))
    #print(cluster_profile_data)
    dfs[8]<-cluster_profile_data[1]
    # 
    dfs[9]<-cluster_profile_data[2]
    dfs[10]<-cluster_profile_data[3];
    print("======== Done clustering profiling ========")
    dfs[11]<-cluster_profile_data[4]
    dfs[12]<-cluster_profile_data[5]
    dbDisconnect(con);
    remove(con);
    dfs
  }else{
    print("There is No correlation data")
  }
}

#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param yValue
#* @param xValues
#* @param controlParameter
#* @param fitParameter
#* @param columnsArray
#* @get /decisiontree
decisiontree<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",yValue="",xValues="",controlParameter="",fitParameter="",columnsArray="") {
  lapply(list('jsonlite','partykit','rattle','rpart.plot','rpart','caret','jsonlite','devtools','ps','aCRM','DBI','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx','MLmetrics'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  h2o.init(nthreads=-1,enable_assertions = FALSE)
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  columnsArray <- fromJSON(columnsArray)
  y=c(fromJSON(yValue))
  # for(j in 1:length(row.names(columnsArray))){
  #   if(y == (columnsArray["columnName"][j,1])){
  #     y <- columnsArray["columnDisplayName"][j,1]
  #   }
  # }
  y<-gsub(x=y,pattern = "\\ ",replacement = "_")
  y<-gsub("[^[:alnum:]///' ]", "", y)
  changed_data <- columns_data_type(data,columnsArray)
  correlationData<-correlations(changed_data)
  complete_data<-correlationData[3]
  new_complete_data<-complete_data[[1]]
  # for(i in 1:length(names(new_complete_data))){
  #   for(j in 1:length(row.names(columnsArray))){
  #     if(names(new_complete_data)[i] == (columnsArray["columnName"][j,1])){
  #       names(new_complete_data)[i] <- columnsArray["columnDisplayName"][j,1]
  #     }
  #     if(names(new_complete_data)[i] == (columnsArray["columnName"][j,1])){
  #       names(new_complete_data)[i] <- columnsArray["columnDisplayName"][j,1]
  #     }
  #   }
  # }
  features<-names(new_complete_data)
  features <- gsub(x = features,pattern = "\\ ",replacement = "_")
  features<-gsub("[^[:alnum:]///' ]", "", features)
  names(new_complete_data)<-features
  #features <- gsub(x = features,pattern = "\\ ",replacement = "_")
  dfs<-decision_trees(new_complete_data,y,controlParameter,features,fitParameter)
  print("======== Done Decision Tree ========")
  dbDisconnect(con);
  remove(con);
  dfs
}

#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param textColumns
#* @param columnsArray
#* @get /co_occurrence_graph
co_occurrence_graph<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",textColumns="",columnsArray="") {
  lapply(list('jsonlite','qdap','widyr','ggraph','igraph','wordcloud','tidytext','tidyverse','tm','devtools','ps','DBI','aCRM','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  h2o.init(nthreads=-1,enable_assertions = FALSE)
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  columnsArray <- fromJSON(columnsArray)
  print(textColumns)
  text=c(fromJSON(textColumns))
  #text<-gsub(x=text,pattern = "\\ ",replacement = "_")
  data<-na.omit(data)
  df<-subset(data,select=c(text))
  finaldata<-lapply(df, co_occurrence)
  for(i in 1:length(names(finaldata))){
    for(j in 1:length(row.names(columnsArray))){
      if(names(finaldata)[i] == (columnsArray["changedColumnName"][j,1])){
        names(finaldata)[i] <- columnsArray["columnDisplayName"][j,1]
      }
    }
  }
  dfs<-list(finaldata)
  print("======== Done co_occurrence graph ========")
  dbDisconnect(con);
  remove(con);
  dfs
}

#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param yValue
#* @param xValues
#* @param dateColumns
#* @param parametersObj
#* @param columnsArray
#* @get /forecasting
forecasting<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",yValue="",xValues="",dateColumns="",parametersObj="",columnsArray="") {
  lapply(list('jsonlite','DMwR','gtools','forecast','highcharter','dplyr','xts','zoo','lubridate','devtools','ps','DBI','aCRM','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx'), require, character.only = TRUE)
  print("======== Started forecasting ========")
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  h2o.init(nthreads=-1,enable_assertions = FALSE)
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  dateValues=c(fromJSON(dateColumns))
  features<-fromJSON(xValues);
  y=c(fromJSON(yValue))
  columnsArray <- fromJSON(columnsArray)
  yDisplay <- y
  for(i in 1:length(row.names(columnsArray))){
    if(y == (columnsArray["changedColumnName"][i,1])){
      yDisplay <- columnsArray["columnDisplayName"][i,1]
    }
  }
  #features <- gsub(x = features,pattern = "\\ ",replacement = "_")
  #y<-gsub(x=y,pattern = "\\ ",replacement = "_")
  #dateValues <- gsub(x = dateValues,pattern = "\\ ",replacement = "_")
  parameters="None"
  if(tolower(fromJSON(parametersObj))!="none"){
    parameters<-fromJSON(parametersObj);
  } 
  print(parameters$no_of_periods_to_forecast)
  print(c(parameters$independentVariables))
  print(parameters$futureValues)
  data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)],as.factor)
  data = knnImputation(data)
  data[,dateValues] = as.Date(data[,dateValues])
  dfs<-forecasting_model(data,y,dateValues,parameters$forecastingfamily,c(parameters$no_of_periods_to_forecast),c(parameters$independentVariables),c(parameters$futureValues),yDisplay)
  print("======== Done forecasting ========")
  dbDisconnect(con);
  remove(con);
  dfs
}



#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param yValue
#* @param xValues
#* @param parametersObj
#* @param columnsArray
#* @get /logistic_regression
logisticregression<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",yValue="",xValues="",parametersObj="",columnsArray="") {
  lapply(list('jsonlite','nnet','DMwR','gtools','forecast','highcharter','dplyr','xts','zoo','lubridate','devtools','ps','DBI','aCRM','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  h2o.init(nthreads=-1,enable_assertions = FALSE)
  y=c(fromJSON(yValue))
  y<-gsub(x=y,pattern = "\\ ",replacement = "_")
  y<-gsub("[^[:alnum:]///' ]", "", y)
  features<-fromJSON(xValues);
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  columnsArray <- fromJSON(columnsArray)
  changed_data <- columns_data_type(data,columnsArray)
  correlationData<-correlations(changed_data)
  complete_data<-correlationData[3]
  new_complete_data<-complete_data[[1]]
  features<-names(new_complete_data)
  features <- gsub(x = features,pattern = "\\ ",replacement = "_")
  features<-gsub("[^[:alnum:]///' ]", "", features)
  names(new_complete_data)<-features
  new<-transformation(new_complete_data,features)
  vars<-variable_importance_h2o(new,y)
  dfs<-modelling_GBM(new_complete_data,y,parameters="None",features,vars)
  print("======== Logistic Regression ========")
  dbDisconnect(con);
  remove(con);
  dfs
}

#* @param dbHost 
#* @param dbPort 
#* @param userName 
#* @param password
#* @param dbName 
#* @param query
#* @param textColumns
#* @param parametersObj
#* @param columnsArray
#* @get /text_analysis
text_analysis<-function(dbHost="",dbPort="",userName="",password="",dbName="",query="",textColumns="",parametersObj="",columnsArray="") {
  lapply(list('jsonlite','tidyverse','tidytext','topicmodels','tm','SnowballC','dplyr','ldatuning','topicmodels','qdap','stringr','devtools','DBI','h20','tm','SnowballC','wordcloud','RColorBrewer'), require, character.only = TRUE)
  con <- dbConnect(clickhouse::clickhouse(), host=dbHost, port=dbPort, user=userName, password=password,db=dbName)
  h2o.init(nthreads=-1,enable_assertions = FALSE)
  data <-dbGetQuery(con, query);
  data<-as.data.frame(data);
  print(textColumns)
  Text=c(fromJSON(textColumns))
  #text<-gsub(x=text,pattern = "\\ ",replacement = "_")
  #data<-na.omit(data)
  parameters="None"
  if(tolower(fromJSON(parametersObj))!="none"){
    parameters<-fromJSON(parametersObj);
  } 
  #df<-subset(data,select=c(text))
  #finaldata<-lapply(df, co_occurrence)
  #dfs<-list(finaldata)
  #print(corpus)
  burnin <- 4000
  iter <- 2000
  thin <- 500
  nstart <- 5
  best <- TRUE
  seed <- list(2003,5,63,100001,765)
  #topp<-top_terms_by_topic_LDA(reviews,Text,k = 4,nstart,best,burnin,seed)
  #corpus=clean_text1(reviews,Text)
  # djfd=sentiment(data=reviews,Text=Text,topic=1,ldaOut.terms=topp[2],corpus = corpus)
  
  topp<-top_terms_by_topic_LDA(reviews,Text,k = 4,nstart,best,burnin,seed)
  corpus=clean_text1(reviews,Text)
  djfd=sentiment(data=reviews,Text=Text,topic=1,ldaOut.terms=topp[2],corpus = corpus)
  print("======== Done sentiment ========")
  dbDisconnect(con);
  remove(con);
  djfd
}