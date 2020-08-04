
#### Unique_data
#test<-read.csv("/users/navyarao/Downloads/bankempfilter1.csv")
#test<-test[,-1]
#k1<-read.csv("/users/navyarao/Downloads/bankempcoff1.csv")
#k1<-k1[,-1]
#data1 = subset_data
data_str<-function(data1){
  data_unique<-data.frame()
  nums<-names(select_if(data1,is.numeric))
  numeric_data<-data.frame(data1[ ,nums])
  names(numeric_data)=nums
  cats<-names(select_if(data1,is.factor))
  categorical_data<-data.frame(data1[ ,cats])
  names(categorical_data)=cats
  if(length(categorical_data)>=1){
    for(i in 1:ncol(categorical_data)){
      levels_data<-data.frame("var_name"=colnames(categorical_data[i]),"levels"=c(levels(categorical_data[,i])))
      data_unique<-rbind(data_unique,levels_data)
    }
    levels_num_data<-data.frame(numeric_data[sample(nrow(numeric_data), 1), ])
    names(levels_num_data)=nums
    num_level<-data.frame(t(levels_num_data))
    dat<-setDT(num_level, keep.rownames = TRUE)[]
    names(dat)[1]<-"var_name"
    names(dat)[2]<-"levels"
    data_unique<-rbind(data_unique,dat)
  }else{
    dat<-data.frame(t(data.frame(numeric_data[1,])))
    dat$var_name<-rownames(dat)
    names(dat)[1]<-"levels"
    data_unique<-dat
    data_unique <- data_unique[c("var_name", "levels")]
    rownames(data_unique) <- NULL
  }
  return(data_unique)
}

#select=h2o.coef.model.gaussian.
# test= subset_data[,-c(ncol(subset_data))]
# 
# k1 = j[4][[1]]
simulation<-function(test,k1){
  new<-data.frame()
  teta1<-setNames(data.frame(matrix(ncol = 1, nrow = 0)), c("h2o.coef.model.gaussian."))
  names(test)=gsub(x = names(test),pattern = "[^[:alnum:]]",replacement = "_")
   for(i in names(test)){
    
    if(i%in%k1$left){
      print(i)
      if(class(test[,which(names(test)==i)])=="factor"){
        print("class factor")
        if(test[,i]%in%k1$right){
          selected<-test[,i]
          subsetted<-k1[k1$right %in% selected,]
          teta1<-data.frame(subsetted$h2o.coef.model.gaussian.)
        }else{
          teta1<- data.frame("h2o.coef.model.gaussian." = 0)
        }
      }else{
        print("else")
        teta1<-subset(k1,right==i,select=h2o.coef.model.gaussian.)
        val<-test[,i]
        teta1<-data.frame("h2o.coef.model.gaussian."=c(teta1$h2o.coef.model.gaussian.*val))
        #new_val<-teta1
      }
      names(teta1)[1]<-"h2o.coef.model.gaussian."
      new<-rbind(new,teta1)
      names(new)[1]<-"h2o.coef.model.gaussian."
    }else{
      teta1<- data.frame("h2o.coef.model.gaussian." = 0)
      if(nrow(new) >0 ){
        names(new)[1]<-"h2o.coef.model.gaussian."
        new<-rbind(new,teta1)
      }else{
        new =data.frame("h2o.coef.model.gaussian." = 0)
      }

    }
  }
  intercept<-subset(k1,right=="Intercept",select=h2o.coef.model.gaussian.)
  names(intercept)[1]<-"h2o.coef.model.gaussian."
  new<-rbind(new,intercept)
  simulated_value<-sum(new$h2o.coef.model.gaussian.)
  listed<-list(simulated_value,new,intercept)
  return(listed)
}  




