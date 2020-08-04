correlations<-function(df){
  #df<-subset(df,select=c(features,y))
  nam_int=names(df)
  df <- df %>% select(-contains("ID"))
  is.na(df) <- df==''##to replace blanks
  library(tidyverse)
  missing_vals<-map(df, ~mean(is.na(.))) 
  missing_values<-as.data.frame(t(as.data.frame(missing_vals)))
  missing_values$columns<-rownames(missing_values)
  missing_drop = missing_values$V1 > 0.4
  if(sum(missing_drop)> 0){
    df = df[,-c(which(missing_drop))]
  }
  #df<-df[,colSums(is.na(df))<nrow(df)]
  dataframe<-data.frame(findAndTransformDates(df, cols = "auto", formats = NULL,n_test = 30, ambiguities = "IGNORE", verbose = TRUE))
  dataframe <- japply(dataframe, which(sapply(dataframe, class)=="integer"), as.numeric )
  dataframe[sapply(dataframe, is.logical)] <- lapply(dataframe[sapply(dataframe, is.logical)],as.factor)
  names(dataframe)<-names(df)
  nums<-names(select_if(dataframe,is.numeric))
  numeric_data<-data.frame(dataframe[ ,nums])
  names(numeric_data)<-nums
  #dataframe <- japply( dataframe, which(sapply(dataframe, class)=="character"), as.factor)
  dataframe[sapply(dataframe, is.character)] <- lapply(dataframe[sapply(dataframe, is.character)],as.factor)
  cat<-names(select_if(dataframe,is.factor))
  missing_values<-sum(is.na(dataframe))/prod(dim(dataframe))
  if(missing_values<0.4){
    if(length(numeric_data)>=1){
      numeric_data[sapply(numeric_data, is.numeric)] <- lapply(numeric_data[sapply(numeric_data, is.numeric)], function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))
    }
    combine_data<-numeric_data
    if(length(cat)>=1){
      cat_data<-dataframe[ , cat]
      cat_data<-data.frame(cat_data)
      names(cat_data)<-cat
      setDT(cat_data)
      dt<-data.frame(lapply(cat_data,function(x) ((length(unique(x))/nrow(cat_data)))*100))
      todrop<-apply(dt, 1, function(x) colnames(dt)[which(x >= 30)])
      print("Removing variables that contains more than 500 levels")
      cats<-cat_data[, (todrop) := NULL]
      cat_data<-data.frame(cats)
      names(cat_data)<-names(cats)
      cat_data<-imputeMissings(cat_data)
      names(cat_data)<-names(cats)
      if(length(cats)>=1){
        cat_data<-data.frame(cats)
        names(cat_data)<-names(cats)
        cat_data<-imputeMissings(cat_data)
        names(cat_data)<-names(cats)
        combine_data<-cbind(cat_data,numeric_data)
      }else{
        combine_data<-numeric_data
      }
    }
  }else{
    combine_data<-na.omit(dataframe)
    nums<-names(select_if(combine_data,is.numeric))
    numeric_data<-data.frame(combine_data[ , nums])
    names(numeric_data)<-nums
    cat<-names(select_if(combine_data,is.factor))
    if(length(cat)>=1){
      cat_data<-data.frame(combine_data[ , cat])
      names(cat_data)<-cat
      setDT(cat_data)
      dt<-data.frame(lapply(cat_data,function(x) ((length(unique(x))/nrow(cat_data))*100)))
      todrop<-apply(dt, 1, function(x) colnames(dt)[which(x >= 3)])
      cats<-cat_data[, (todrop) := NULL]
      names_cats<-names(cats)
      cat_data<-data.frame(cats)
      if(nrow(cat_data)>=1){
        names(cat_data)<-names_cats
        combine_data<-cbind(cat_data,numeric_data)
      }else{
        combine_data<-numeric_data
      }
    }
  }
  logic<-sapply(combine_data, function(v) var(v, na.rm=TRUE)!=0)
  nam_logic<-which(logic==TRUE)
  new_combine_data<-as.data.frame(combine_data[,sapply(combine_data, function(v) var(v, na.rm=TRUE)!=0)])
  nums<-names(select_if(new_combine_data,is.numeric))
  numeric_data<-as.data.frame(new_combine_data[ ,nums])
  names(numeric_data)<-nums
  cat<-names(select_if(new_combine_data,is.factor))
  if(length(cat)>=1){
    cat_data<-new_combine_data[ , cat]
    cat_data<-as.data.frame(cat_data)
    names(cat_data)<-cat
    setDT(cat_data)
    dt<-as.data.frame(lapply(cat_data,function(x) ((length(unique(x))/nrow(cat_data))*100)))
    names(dt)<-names(cat_data)
    todrop<-apply(dt, 1, function(x) colnames(dt)[which(x >= 3)])
    cats<-cat_data[, (todrop) := NULL]
    cat_data<-as.data.frame(cats)
    if(nrow(cat_data)>=1){
      cat_data<-as.data.frame(cat_data)
      new_combine_data<-cbind(cat_data,numeric_data)
    }else{
      new_combine_data<-numeric_data
    }
  }
  numeric_data<-as.data.frame(numeric_data)
  names(numeric_data)<-nums
  ent_cor<-data.frame(cor(numeric_data))
  names(ent_cor)<-nums
  ent_cor[is.na(ent_cor)]<-0
  ent_cor1<-abs(ent_cor)
  ent_cor1<-t(apply(ent_cor1, 1, function(x) replace(x, x== max(x), 0)))
  for(i in 1:nrow(ent_cor1)){
    maxi<-max(ent_cor1[i,])
    ent_cor1[i,][ent_cor1[i,]< maxi]<-0
  }
  if(length(ent_cor1)>1){
    max_cor_info<-data.frame(x=rownames(ent_cor1),y=colnames(ent_cor1)[apply(ent_cor1,1,which.max)],coefficient=apply(ent_cor1[, 1:nrow(ent_cor1)], 1, max))
    rownames(max_cor_info)<-NULL
    dependencies <- as.data.table(max_cor_info)
    targets<-data.frame(setDT(dependencies)[, .SD[which.min(coefficient)], by=x])
    k<-tail(names(sort(table(targets$y))), 1)
    most_rep<-subset(targets,y==k)
    target_var<-unique(most_rep$y)
  }else{
    target_var<-names(numeric_data)
  }
  listed<-list()
  x<-list()
  y<-list()
  df_ano<-data.frame()
  numeric_data<-as.data.frame(numeric_data)
  cat<-names(select_if(new_combine_data,is.factor))
  if(length(cat)>=1&length(numeric_data)>=1){
    start.time <- Sys.time()
    for(i in 1:ncol(numeric_data)){
      for(j in 1:ncol(cat_data)){
        c =data.table(unlist(summary(aov(numeric_data[,i] ~ cat_data[,j]))))
        #new<-data.table("p-value"=c["Pr(>F)1",])
        x<-data.table("y"=names(numeric_data[i]))
        y<-data.table("x"=names(cat_data[j]))
        anov_res<-cbind(x,y,c)
        df_ano<-rbind(df_ano,anov_res)
      }
    }
    dependents<-subset(df_ano,V1 < 0.05)
    dependents <- as.data.table(dependents)
    depend<-data.frame(setDT(dependents)[, .SD[which.min(V1)], by=y])
    dependent_num<-depend[which.max(depend$V1),]
    dependent_variable_num<-data.frame(rbind(dependent_num$y,target_var))
    if(ncol(cat_data)>1){
      chissq_dependency<-ch_sq_test(cat_data)
      tt <- table(chissq_dependency$Column)
      cat_dependent<-names(tt[which.max(tt)])
    }else{
      cat_dependent<-(names(cat_data))
    }
    missing_columns<-setdiff(nam_int,names(new_combine_data))
    print(paste0("missing_columns in Fun_corr------",missing_columns))
    #names(new_combine_data) <- gsub(x = names(new_combine_data),pattern = "\\.",replacement = "_")
    feature_eng_cols<-names(new_combine_data)
    listed<-list(target_var,ent_cor,new_combine_data,combine_data,cat_data,missing_values,missing_columns,feature_eng_cols)
    #return(listed)
  }else{
    #names(new_combine_data) <- gsub(x = names(new_combine_data),pattern = "\\.",replacement = "_")
    missing_cols<-setdiff(names(new_combine_data),nam_int)
    feature_eng_cols<-names(new_combine_data)
    print(paste0("missing_columns in Fun_corr------",missing_cols))
    listed<-list(target_var,ent_cor,new_combine_data,combine_data,"no categorical data",missing_values,missing_cols,feature_eng_cols)
  }
  #return(listed)
}