
NLG<-function(data,var,Target){
  out <- lapply(data, function(x) length(unique(x)))
  want <- data.frame(unlist(which(!out > 1)))
  if(nrow(want)>=1){
    data<-data.frame(data[,-which(names(data)%in%rownames(want))])
  }else{
    data<-data.frame(data)
  }
  names(data) <- gsub(x = names(data),pattern = "\\.",replacement = "_")
  nums<-names(select_if(data,is.numeric))
  numeric_data<-data[ ,nums]
  cats<-names(select_if(data,is.factor))
  cat_data<-data[ ,cats]
  cor_mat<-cor(numeric_data)
  ent_cor<-t(apply(cor_mat, 1, function(x) replace(x, x== max(x), 0)))
  
  if(class(data[,which(names(data)==var)])!="factor"){
    corr<-cor(data[,var],data[,Target])
    if(corr>=0.3){
      cor_val<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between <strong>" ,var, "</strong><strong>", Target, "</strong>, With every unit of increase in", var, "there is an increase in" ,Target,"</span></li>")
    }
    if(corr<= -0.2){
      cor_val<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\">There is a mutual relationship between<strong> " ,var, "</strong>and <strong>", Target, "</strong></span>, With every unit of decrease in", var, "there is an increase in" ,Target,"</span></li>")
    }
    if(corr<=0.3|corr>=0.2){
      cor_val<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\">There is a little or no relationship between <strong>",var,"</strong> and <strong>",Target,"</strong></span></li>")
    }
    avg<-mean(data[,var])
    var_val<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\">The average value of <strong>",var,"</strong> is <strong>",round(avg,digits = 3),"</strong></span>","</li>")
    quantiles<-quantile(data[,var])
    quantile_val<-paste("<li class=\"nlg-item\"><span class=\"nlgtext\">The minimum value of <strong>",var,"</strong> is <strong>" ,round(quantiles[[1]],digits = 3),"</strong> whereas <strong>25%</strong> of data lies below the value <strong>",round(quantiles[[2]],digits = 3), "</strong> the median of <strong>",var, "</strong> is <strong>",round(quantiles[[3]],digits = 3),"</strong> and the <strong>75%</strong> of data lies below the value <strong>",round(quantiles[[3]],digits = 3),"</strong> and the max value is <strong>",round(quantiles[[5]],digits = 3),"</strong></span></li>")
    skew<-skewness(data[,which(names(data)==var)])
    if(skew>avg){
      skew_val<-paste("<li class=\"nlg-item\"> <span class=\"nlgtext\">Most of the data lies above the average value of <strong>",var,"</strong></span>","</li>")
    }else if(skew < avg){
      skew_val<-paste("<li class=\"nlg-item\"> <span class=\"nlgtext\">Most of the data lies below the average value of <strong>",var,"</strong></span>","</li>")
    }else{
      skew_val<-paste("<li class=\"nlg-item\"> <span class=\"nlgtext\">Most of the data lies around the average value of <strong>",var,"</strong></span>","</li>")
    }
    listed<-list(cor_val,var_val,quantile_val,skew_val)
    return(listed)
  }
  if(class(data[,which(names(data)==var)])=="factor"){
    w = as.data.frame(table(data[,var]))
    names(w)[1]<-"Values"
    names(w)[2]<-"Frequency"
    data_sorted<-tail(w[order( w[,2] ),])
    mean_target<-mean(data[,Target])
    avg_tar<-paste("The average of <strong>",Target,"</strong> is",round(mean_target,digits = 3))
    temp<-subset(data,select=c(var,Target))
    Agg<-aggregate(temp[,Target], by=list(temp[,var]), FUN=sum)
    names(Agg)[1]<-"Values"
    names(Agg)[2]<-"Frequency"
    max_val<-Agg[which.max(Agg$Frequency),]
    min_val<-Agg[which.min(Agg$Frequency),]
    max_agg_val<-paste("<li class=\"nlg-item\"> <span class=\"nlgtext\">The maximum <strong>",Target,"</strong> corresponds to <strong>",max_val$Values,"</strong> which is <strong>",round(max_val$Frequency,digits = 3),"</strong> whereas the frequency of <strong>",max_val$Values,"</strong> in the data is <strong>",subset(w,Values==max_val$Values,select=Frequency),"</strong></span></li>")
    min_agg_val<-paste("<li class=\"nlg-item\"> <span class=\"nlgtext\">The minimum <strong>",Target,"</strong> corresponds to <strong>",min_val$Values,"</strong> which is <strong>",round(min_val$Frequency,digits = 3),"</strong> whereas the frequency of <strong>",min_val$Values,"</strong> in the data is <strong>",subset(w,Values==min_val$Values,select=Frequency),"</strong></span></li>")
    cat_data=data.frame(cat_data)
    ch<-ch_sq_test(cat_data) 
    ch_val<-""
    if(!is.data.frame(ch) && ch=="There is only one category field in dataframe"){
      ch_val="There is only one category field in dataframe"
    }else{
    if(var%in%ch$Row){
      eac<-subset(ch,Row==var,select=Column)
      if(length(eac>1)){
        vec<-eac$Column
        ch_val<-paste("<li class=\"nlg-item\"> <span class=\"nlgtext\">There is a relationship associated between <Strong>",var,"</strong> and <strong>",eac$Column[1:length(vec)],"</strong></span></li>")
      }else{
        ch_val<-paste("<li class=\"nlg-item\"> <span class=\"nlgtext\">There is a relationship associated between <strong>",var,eac$Column,"</strong></span></li>")
      }
    }
    }
  }
  listed<-list(avg_tar,max_agg_val,min_agg_val,ch_val)
  return(listed)
}

