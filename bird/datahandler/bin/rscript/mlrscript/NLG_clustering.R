cluster_nlg<-function(withiness_tab,betweeness){
  withiness_tab$cluster<-c(1:nrow(withiness_tab))
  mini<-withiness_tab[which.min(withiness_tab$withiness),]
  withiness_desc<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\"> <strong> Cluster", mini$cluster, "</strong> has lowest value of withiness. The points within the <strong> Cluster", mini$cluster, "</strong> are more homogenous </span></li>" )
  if(betweeness>=10000){
    betweeness_desc<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have reasonably high value of betweeness  <strong>",round(betweeness),"  </strong> indicating that the heterogeneity among them is high </span></li>")
  }else if(betweeness>5000&betweeness<10000){
    betweeness_desc<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have medium level of betweeness  <strong>",round(betweeness)," </strong>indicating that the heterogeneity among them is not very high </span></li>")
  }else{
      if(betweeness<5000){ 
        betweeness_desc<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\"> Clusters have low level of betweeness <strong>",round(betweeness),"  </strong> indicating that the heterogeneity among them is low </span></li>")
      }
  }
  listed<-list(withiness_desc,betweeness_desc)
  return(listed)
}

profiling_nlg<-function(data_num,data_cat){
  uni<-unique(data_num$var_name)
  maxmeans_num<-list()
  minmean_num<-list()
  cat_info<-list()
  if(length(uni)>0){
    for(i in 1:length(uni)){
      sub<-subset(data_num,var_name==uni[i])
      sub_mean<-sub[which.max(sub$means),]
      max_mean<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\"> The mean value of  <strong>",sub_mean$var_name ,"</strong> in <strong> cluster",sub_mean$cluster,"</strong> is highest across other <strong>",length(unique(sub$cluster)),"</strong> clusters","</span></li>")
      maxmeans_num<-c(maxmeans_num,max_mean)
      sub_mean_min<-sub[which.min(sub$means),]
      min_mean<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\"> The mean value of  <strong>",sub_mean_min$var_name,"</strong> in <strong> cluster",sub_mean_min$cluster,"</strong> is the lowest across other <strong>",length(unique(sub$cluster)),"</strong> clusters","</span></li>")
      minmean_num<-c(minmean_num,min_mean)
    }
  }
  uni_cat<-unique(data_cat$var_name)
  if(length(uni_cat)>0){
    for(i in 1:length(uni_cat)){
      sub_cat<-subset(data_cat,var_name==uni_cat[i])
      for(i in 1:length(unique(sub_cat$cluster))){
        sub_lev<-subset(sub_cat,cluster==i)
        sub_frq<-sub_lev[which.max(sub_lev$value),]
        freq_info<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\"> The frequency of <strong>",sub_frq$var_name, "</strong> - <strong>",sub_frq$name,"</strong> is the most repeated level in <strong> Cluster",sub_frq$cluster,"</strong></span></li>")
        cat_info<-c(cat_info,freq_info)
      }
    }
  }
  listed<-list(maxmeans_num,minmean_num,cat_info)
 return(listed) 
}
