#### Variable_importance(Key influencers)
options(warn=-1)
variable_importance_h2o<-function(dataframe,variable){
  if(ncol(dataframe)>2){
  names(dataframe) <- gsub(x = names(dataframe),pattern = "\\.",replacement = "_")
  pos<-names(dplyr::select_if(dataframe,is.POSIXct))
  if(length(pos)>=1){
    dats<-dataframe[ , -which(names(dataframe) %in% c(pos))]
  }else{
    dats<-dataframe
  }
  names(dats) <- gsub(x = names(dats),pattern = "\\.",replacement = "__")
  df.hex <- as.h2o(dats)
  if(class(dataframe[,which(names(dataframe)==variable)])!="factor"){
    print("Finding variable importance by taking given numeric variable as a dependent variable")
    prostate.glm<-h2o.glm(y = variable, x = names(dats[,-which(names(dats)==variable)]), training_frame = df.hex, alpha =0.5)
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
    var_imp2<-data.frame(variable=var_imp2$variable,coefficients=var_imp2$scaled_importance)
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
    return(Fin_imp_var)
  }else{
    print("Finding variable importance by taking categorical variables as dependent variable")
    df.gbm <- h2o.gbm(y = variable, x = names(dats[,-which(names(dats)==variable)]), training_frame =df.hex, ntrees = 5,max_depth = 3,min_rows = 2, learn_rate = 0.2)
    var_imp2<-data.frame(h2o.varimp(df.gbm))
    var_imp2<-data.frame(variable=var_imp2$variable,coefficients=var_imp2$scaled_importance)
    Fin_imp_var<-subset(var_imp2,coefficients>0,select=c(variable,coefficients))
    return(Fin_imp_var)
  }
  }else{
    Error<-("Cannot calculate variable importance if the variables are less than 2")
    return(Error)
  }
}
