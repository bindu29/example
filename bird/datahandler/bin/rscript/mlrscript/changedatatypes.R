columns_data_type<-function(df,columnsArray=""){
  columnsArray<-as.data.frame(columnsArray)
  names(df)<-columnsArray$changedColumnName
  list_names<-columnsArray$changedColumnName
  for(i in columnsArray$changedColumnName){
    sub<-subset(columnsArray,changedColumnName==i)
    if(sub[,"tableDisplayType"]=="number"){
      df[,i]<-as.numeric(df[,i])
    }else if(sub[,"tableDisplayType"]=="string"){
      df[,i]<-as.factor(df[,i])
    }
  }
  return(df)
}
