
transformation<-function(comb_data){
  names(comb_data) <- gsub(x = names(comb_data),pattern = "\\.",replacement = "_")
  nums<-names(select_if(comb_data,is.numeric))
  numeric_data<-data.frame(comb_data[,nums])
  fact<-names(select_if(comb_data,is.factor))
  if(length(fact)>=1){
    cat_data<-data.frame(comb_data[ ,fact])
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
