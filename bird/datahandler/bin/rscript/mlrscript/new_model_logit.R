modelling_GBM<-function(df,y,parameters="None",features,vars){
  #df<-subset(df,select=c(features))
  set.seed(101)
  sample <- sample.int(n = nrow(df), size = floor(.75*nrow(df)), replace = F)
  df[,y]<-as.factor(df[,y])
  train <- df[sample, ]
  train[,y]<-as.factor(train[,y])
  test  <- df[-sample, ]
  test[,y]<-as.factor(test[,y])
  dats_test<-test[,-which(names(test)==y)]
  Actual<-test[,y]
  test[,y]<-gsub("[^0-9A-Za-z///' ]", "", test[,y])
  test[,y]<-as.factor(test[,y])
  levs<-length(unique(Actual))
  if(parameters=="None"){
    if(levs==2){
      #model.fit<-glm.fit(x=train[,-which(names(train)==y)],y=y,family="binomial"(link = "logit"))
      df[,y]<-as.factor(y)
      form=as.formula(paste(y,"~."))
      model.fit<-glm(form,data=train,family="binomial"(link = 'logit'))
    }else{
      df[,y]<-as.factor(y)
      form=as.formula(paste(y,"~."))
      model.fit<-multinom(form, data = train)
    }
  }else{
    if(levs==2){
      df[,y]<-as.factor(y)
      form=as.formula(paste(y,"~."))
      model.fit<-glm(form,data=train,family="binomial"(link = ''))
    }else{
      df[,y]<-as.factor(y)
      form=as.formula(paste(y,"~."))
      model.fit<-multinom(form, data = train)
    }
  }
  model<-model.fit
  model.fit$xlevels[[y]] <- union(model.fit$xlevels[[as.factor(y)]], levels(test[,y]))
  yhat <- data.frame("probability"=predict(model.fit, dats_test, type="response"))
  yhat$class<-round(yhat$probability)
  dataframe<-cbind(dats_test,yhat)
  dataframe$Actual<-Actual
  names(dataframe)[names(dataframe) == 'class'] <- 'Predicted_values'
  cm=confusionMatrix(as.factor(dataframe$Predicted_values),as.factor(dataframe$Actual))
  cm_table<-as.data.frame(cm$table)
  accuracy<-round(cm$overall[1]*10)
  tab<-cm$table
  precision <- data.frame("precision"=(diag(tab) / rowSums(tab)))
  precision$levels<-rownames(precision)
  recall <-data.frame("recall"= (diag(tab) / colSums(tab)))
  recall$levels<-rownames(recall)
  listed<-list(dataframe,accuracy,vars,cm_table)
  return(listed)
}