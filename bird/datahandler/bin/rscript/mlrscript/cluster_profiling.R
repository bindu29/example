# data<-read.csv("/users/navyarao/Downloads/clustermodeloutput.csv")
# data<-data[,-c(1,2)]
# data<-data[c(1:100),]

cluster_profiling<-function(data){ 
  # library(Boruta)
  vars <- variable_importance_h2o(data,"cluster")
  important_vars<-vars$x
  important_vars<-head(important_vars,6)
  imp_data <- data[,which(colnames(data)%in%important_vars)]
  imp_data[,"cluster"]<-data[,"cluster"]
  num_cluster<-length(unique(imp_data$cluster))
  mean_num<-data.frame()
  median_num<-data.frame()
  frq<-data.frame()
  num_list<-list()
  cat_list<-list()
  headersobj<-data.frame()
  for(i in 1:num_cluster){
    print(i)
    test_i= split(imp_data,imp_data[,"cluster"])[[i]]
    
    headersobj_cat<-data.frame("id"=paste("cluster",i),"name"=paste("Cluster",i))
    headersobj<-rbind(headersobj,headersobj_cat)
    for(j in 1:ncol(test_i)){
      #print(j)
      if(class(test_i[,j])=="numeric"){
        mean_var<-data.frame("var_name"=names(test_i)[j],"means"=mean(test_i[,j]),"cluster"=i,"median"=median(test_i[,j]))
        mean_num<-rbind(mean_num,mean_var)
        if(length(mean_num)<1){
          mean_num<-0
        }
      }else{
        
        freq_cat<-data.frame("var_name"=names(test_i)[j],"levels"=table(test_i[,j]),"cluster"=i,"parent"=paste("cluster",i))
        frq<-rbind(frq,freq_cat)
        if(length(frq)<1){
          frq<-0
        }
        
        frq=frq[!grepl("cluster", frq$var_name),]
        #listed(clust,frq)
      }
    }
  }
  if(length(mean_num)>0){
    for (k in 1:length(split(mean_num,mean_num[,"var_name"]))) {
      if(nrow(split(mean_num,mean_num[,"var_name"])[[k]])>0){
        num_list[k] = list(split(mean_num,mean_num[,"var_name"])[[k]])
      }
    }
  }
  if(length(frq)>0){
    names(frq)<-c("var_name","name","value","cluster","parent")
     for (k in 1:length(split(frq,frq[,"var_name"]))) {
       if(nrow(split(frq,frq[,"var_name"])[[k]])>0){
         cat_list[k] = list(split(frq,frq[,"var_name"])[[k]])
       }
     }
    mode_lis<-list()
    uniq_clus<-unique(frq$cluster)
    for(k in uniq_clus){
      sub<-subset(frq,cluster==k)
      unique_vars<-unique(sub$var_name)
      for(j in 1:length(unique_vars)){
        new_sub<-subset(sub,var_name==unique_vars[j])
        nl<-list(new_sub[which.max(new_sub$value),])
        mode_lis<-append(mode_lis,nl)
      }
    }
  }
  

  # cat_list[length(split(frq,frq[,"var_name"]))+1] = headersobj
  names(vars)<-c("x","y")
  key_influencer<-head(vars)
  listed<-list((num_list),(cat_list),headersobj,key_influencer,mode_lis)
  return(listed)
}

#a<-cluster_profiling(dfs[1])

#a[1]
