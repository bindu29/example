transformation<-function(comb_data,features){
  if(length(features)>=1){
    features<-names(comb_data)
    comb_data1<-subset(comb_data,select=c(features))
    #comb_data <- comb_data %>% select(-contains("ID"))
  }else{
    comb_data1<-comb_data
  }
  names(comb_data1) <- gsub(x = names(comb_data1),pattern = "\\.",replacement = "_")
  nums<-names(select_if(comb_data1,is.numeric))
  numeric_data<-as.data.frame(comb_data1[,nums])
  names(numeric_data)<-nums
  fact<-names(select_if(comb_data1,is.factor))
  if(length(fact)>=1){
    cat_data<-as.data.frame(comb_data1[ ,fact])
    names(cat_data)<-fact
  }
  preprocessparams <- preProcess(numeric_data, method=c("YeoJohnson"))
  transformed <- predict(preprocessparams, numeric_data)
  if(length(fact)>=1){
    transformed_data<-cbind(cat_data,transformed)
  }else{
    transformed_data<-transformed
  }
  return(transformed_data)
}
