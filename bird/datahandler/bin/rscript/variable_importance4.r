#### Variable_importance(Key influencers)
#tran<-read.csv("/users/navyarao/Downloads/transform_data.csv")
#dataframe = tran
#variable="Male_Patients"
#options(warn=-1)
variable_importance_h2o<-function(dataframe,variable){
  #names(dataframe) <- gsub(x = names(dataframe),pattern = "\\.",replacement = "_")
  #dataframe<-data.frame(findAndTransformDates(dataframe, cols = "auto", formats = NULL,n_test = 30, ambiguities = "IGNORE", verbose = TRUE))
  pos<-names(dplyr::select_if(dataframe,is.POSIXct))
  if(length(pos)>=1){
    dats<-data.frame(dataframe[ , -which(names(dataframe) %in% c(pos))])
  }else{
    dats<-dataframe
  }
  names(dats) <- gsub(x = names(dats),pattern = "\\.",replacement = "_")
  variable <- gsub("\\.","_",variable)
  df.hex <- as.h2o(dats)
  if(class(dataframe[,which(names(dataframe)==variable)])!="factor"){
    print("Finding variable importance by taking given numeric variable as a dependent variable")
  dats<-data.frame(dats)
  prostate.glm<-h2o.glm(y = variable, x = names(dats[,-which(names(dats)==variable)]), training_frame = df.hex)
  var_imp1<-data.frame(h2o.varimp(prostate.glm))
  names(var_imp1)[1]<-"variable"
  names(var_imp1)[2]<-"coefficients"
  var_imps<-data.frame(x=gsub("\\..*","",var_imp1$variable),y=var_imp1$coefficients)
  #var_imps<-data.frame(x=gsub("\\..*","",var_imp1$names),y=var_imp1$coefficients)
  imp_tab <- data.table(var_imps)
  imp_tab<-na.omit(imp_tab)
  new_tab_imp<-imp_tab[,list(y=sum(y)/100), by='x']
  df.gbm <- h2o.gbm(y = variable, x = names(dats[,-which(names(dats)==variable)]), training_frame =df.hex, ntrees = 5,max_depth = 3,min_rows = 2, learn_rate = 0.2,distribution= "gaussian")
  var_imp2<-data.frame(h2o.varimp(df.gbm))
  var_imp2<-data.frame(variable=var_imp2$variable,coefficients=var_imp2$relative_importance)
  new_df<-data.frame()
  for(i in 1:nrow(var_imp2)){
    if(var_imp2$coefficients[i]>new_tab_imp$y[i]){
      imp_var<-data.frame(vars=var_imp2$variable[i])
      new_df<-rbind(new_df,unique(imp_var))
    }else{
      imp_var<-data.frame(vars=new_tab_imp$x[i])
      new_df<-rbind(new_df,unique(imp_var))
    }
    
  }
  names(var_imp2)=names(new_tab_imp)
  comb<-rbind(new_tab_imp,var_imp2)
  sorted<-comb[rev(order(comb$y)),]
  new_sort<-data.frame(setDT(sorted)[, .SD[which.max(y)], by=x])
  Fin_imp_var<-subset(new_sort,y>0,select=c(x,y))
  if(nrow(Fin_imp_var)<=1){
    Info<-c("Since the relative variable importance is only for one variable we are considering all variables")
    Fin_imp_var<-var_imp2
  }
  return(Fin_imp_var)
  }else{
    print("Finding variable importance by taking categorical variables as dependent variable")
    df.gbm <- h2o.gbm(y = variable, x = names(dats[,-which(names(dats)==variable)]), training_frame =df.hex, ntrees = 5,max_depth = 3,min_rows = 2, learn_rate = 0.2)
    var_imp2<-data.frame(h2o.varimp(df.gbm))
    var_imp2<-data.frame(variable=var_imp2$variable,coefficients=var_imp2$scaled_importance)
    Fin_imp_var<-subset(var_imp2,coefficients>0,select=c(variable,coefficients))
    if(nrow(Fin_imp_var)<=1){
      Info<-c("Since the relative variable importance is only for one variable we are considering all variables")
      Fin_imp_var<-var_imp2
    }
    return(Fin_imp_var)
  }
  
  
}

#vars<-variable_importance_h2o(new_dat,"Male_Patients")
#important_vars<-as.character(vars$x)
#new_dat<-data.frame(new_dat)
#subset_data <- new_dat[,which(colnames(new_dat)%in%c(important_vars))]
