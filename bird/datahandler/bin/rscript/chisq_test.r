############ Chisquare test dependency
options(warn=-1)


ch_sq_test<-function(cat_data){
  if(ncol(cat_data)>1){
    combos <- combn(ncol(cat_data),2)
    zx<-apply(combos, 2, function(x) {
      test <- chisq.test(cat_data[, x[1]], cat_data[, x[2]])
      out <- data.frame("Row" = colnames(cat_data)[x[1]]
                        , "Column" = colnames(cat_data[x[2]])
                        , "Chi.Square" = round(test$statistic,3)
                        ,  "df"= test$parameter
                        ,  "p.value" = round(test$p.value, 3))
      
      return(out)
      
    })
    cat_dep <- ldply(zx, data.frame)
    cat_dependency<-subset(cat_dep,p.value<0.05,select=c(Row,Column,Chi.Square))
    return(cat_dependency)
  }else if(ncol(cat_data)==1){
    print("There is only one category field exists")
  }else{
    print("No category field exists")
  }
  
  
}
