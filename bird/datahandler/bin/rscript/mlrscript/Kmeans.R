#Clustering : minimum number of clusters, maximum clusters, nstart
library(fpc)
kmeansmodel<-function(df,parameters="None"){
  df_fin<-df
  #df<-subset(df,select=c(features))
  print(parameters)
  df[sapply(df, is.integer)] <- lapply(df[sapply(df, is.integer)],as.numeric)
  nums<-names(select_if(df,is.numeric))
  numeric_data<-data.frame(df[ ,nums])
  df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)],as.factor)
  cat<-names(select_if(df,is.factor))
  character_data<-data.frame(df[ ,cat])
  scaled_numeric<-scale(numeric_data)
  df<-cbind(character_data,scaled_numeric)
  dummy_dat<- dummy.data.frame(df, sep = ".")
  pr <- prcomp(dummy_dat)
  vars <- apply(pr$x, 2, var)  
  props <- vars / sum(vars)
  props<-data.frame(cumsum(props))
  princ<-subset(props,cumsum.props.<=0.91)
  n_components<-nrow(princ)
  princ_compomp<-data.frame(pr$x)[,c(1:n_components)]
  df.hex <- as.h2o(princ_compomp)
  if(tolower(parameters)=="none"){
    print("with default parameters")
    clustering<-h2o.kmeans(training_frame=df.hex, x=names(princ_compomp),k=5,max_iterations=5,estimate_k = TRUE,seed = 10,categorical_encoding = "AUTO",max_runtime_secs=10)
  }else{
    params_obj<-fromJSON(parameters);
    ignore_const_cols_value = FALSE;
    standardize_value = FALSE;
    estimate_k_value = FALSE;
    score_each_iteration_value = FALSE;
    if(tolower(params_obj$ignore_const_cols)=="true"){
      ignore_const_cols_value=TRUE;
    }
    if(tolower(params_obj$standardize)=="true"){
      standardize_value=TRUE;
    }
    if(tolower(params_obj$score_each_iteration)=="true"){
      score_each_iteration_value = TRUE
    }
    if(tolower(params_obj$estimate_k)=="true"){
      estimate_k_value=TRUE;
    }
    print("with user defined parameters")
    
    clustering<-h2o.kmeans(training_frame=df.hex, x=names(princ_compomp),max_iterations=params_obj$max_iterations,score_each_iteration =score_each_iteration_value,ignore_const_cols=ignore_const_cols_value,
                           k = params_obj$kvalue,max_runtime_secs=params_obj$max_runtime_secs,categorical_encoding=params_obj$categoricalencoding,
                           estimate_k=estimate_k_value,standardize=standardize_value,seed = 1)
  }
  clus<-h2o.predict(clustering,df.hex)
  clust_points<-as.data.frame(clus)
  df_fin$cluster<- clust_points$predict
  dp = discrproj(princ_compomp, clust_points$predict)
  coords<-data.frame(dp$proj[,1], dp$proj[,2])
  names(coords)<-c("PC1","PC2")
  df$cluster<-clust_points$predict
  #df<-cbind(df,coords)
  centroid<-data.frame(clustering@model$centers)
  H2o_tab<-data.frame(clustering@model$model_summary)
  Total_sum_square<-H2o_tab$total_sum_of_squares
  withiness_tab<-data.frame("withiness"=clustering@model$training_metrics@metrics$centroid_stats$within_cluster_sum_of_squares)
  size_cluster<-data.frame("cluster_size"=clustering@model$training_metrics@metrics$centroid_stats$size)
  betweeness<-H2o_tab$between_cluster_sum_of_squares
  listed<-list(df_fin,coords,centroid,withiness_tab,Total_sum_square,betweeness,size_cluster)
  return(listed)
}

#parameters=c(k=20,ignore_const_cols = TRUE, score_each_iteration = FALSE, estimate_k = TRUE,
#             max_iterations = 10, standardize = TRUE, seed = -1, max_runtime_secs = 0,categorical_encoding = "AUTO")
