  #Decision Tree : ( Dependent variable, Independent variable ,method (Anova, Poison, class, exp)
  decision_trees<-function(df,y,features){
    df<-subset(df,select=c(features))
    df[,y]=as.factor(df[,y])
    response =y
    predictors = setdiff(names(df),y) ## Seperate target 
    df_DT = df[, c(response, predictors)]
    df_Hex = as.h2o(df_DT)
    dt_1tree = h2o.gbm(x = predictors, y = response, 
                       training_frame = df_Hex, 
                       ntrees = 1, min_rows = 1, 
                       sample_rate = 1, col_sample_rate_per_tree = 1,
                       max_depth = 4,nfolds=8,learn_rate = 0.01,
                       categorical_encoding = "AUTO",
                       stopping_rounds = 3, stopping_tolerance = 0.01, 
                       stopping_metric = "AUTO",
                       seed = 1)
    var_imp<-data.frame(h2o.varimp(dt_1tree))
    variable_importance<-subset(var_imp,relative_importance>=0.001)
    pred = h2o.predict(dt_1tree, df_Hex)
    predict = as.data.frame(pred$predict)
    df<-cbind(df,predict)
    y<-df[,which(names(df)==y)]
    y=as.factor(y)
    predict_predicted=as.factor(predict)
    cm=confusionMatrix(predict$predict,y)
    cm_table<-as.data.frame(cm$table)
    accuracy<-cm$overall[1]*100
    tab<-cm$table
    precision <- data.frame("precision"=(diag(tab) / rowSums(tab)))
    precision$levels<-rownames(precision)
    recall <-data.frame("recall"= (diag(tab) / colSums(tab)))
    recall$levels<-rownames(recall)
    model_id=dt_1tree@model_id
    tH2oTree = h2o.getModelTree(model = dt_1tree, tree_number =1 )
    DataTree = createDataTree(tH2oTree)
    GetEdgeLabel <- function(node) {return (node$edgeLabel)}
    GetNodeShape <- function(node) {switch(node$type, 
                                           split = "diamond", leaf = "oval")}
    GetFontName <- function(node) {switch(node$type, 
                                          split = 'Palatino-bold', 
                                          leaf = 'Palatino')}
    SetEdgeStyle(DataTree, fontname = 'Palatino-italic', 
                 label = GetEdgeLabel, labelfloat = TRUE,
                 fontsize = "26", fontcolor='royalblue4')
    SetNodeStyle(DataTree, fontname = GetFontName, shape = GetNodeShape, 
                 fontsize = "26", fontcolor='royalblue4',
                 height="0.75", width="1")
    SetGraphStyle(DataTree, rankdir = "LR", dpi=70.)
    plot(DataTree, output = "graph")
    ###tree_info_json = json_prsr(DataTree, node = 1, node_stats = NULL)
    listed<-list("",precision,recall,accuracy,cm_table,variable_importance)
    #listed<-list(tree_info_json,precision,recall,accuracy,cm_table,variable_importance)
    return(listed)
  }
  
  json_prsr <- function(tree_info, node, node_stats){
    # Checking the decision tree object
    if(!is(tree_info, c("constparty","party")))
      tree_info <- partykit::as.party(tree_info)
    
    # Parsing into json format
    jsonstr  <- ""
    rule <- partykit:::.list.rules.party(tree_info, node)
    final_rule <- rule
    final_rule <- str_split(final_rule, " & ")
    final_rule <- final_rule[[1]][length(final_rule[[1]])]
    prob <- predict(tree_info, type = "prob")
    probs_uniq<-data.frame(unique(prob))
    probs_uniq$predicted_node<-colnames(probs_uniq)[apply(probs_uniq,1,which.max)]
    probs_uniq$node_no<-rownames(probs_uniq)
    predicted_node_val<-subset(probs_uniq,node_no==node,select = predicted_node)
    if(nrow(predicted_node_val) == 0){
      predicted_node_val =""
    }else{
      final_rule = ""
      predicted_node_val <- gsub(x = predicted_node_val,pattern = "\\.",replacement = " ")
    }
    if(predicted_node_val =="" & final_rule == "" & node == 1){
      predicted_node_val = "Root"
    }
    if(is.null(node_stats))
      node_stats <- table(tree_info$fitted[1])
    children <- partykit::nodeids(tree_info, node)
    
    if (length(children) == 1) {
      ct  <- node_stats[as.character(children)]
      jsonstr <- paste("{","\"innerHTML\":\"<span id='nodeid ",children,"'", "class='text-center'>", predicted_node_val,"</span><p class='node-name'>",final_rule,"</p>\"","}", sep='')
    } else {
      jsonstr <- paste("{","\"innerHTML\":\"<span id='nodeid ",node,"'", "class='text-center'>", predicted_node_val,"</span><p class='node-name'>",final_rule,"</p>\",\"children\":[", sep='')
      for(child in children){
        check <- paste("{","\"innerHTML\":\"<span id='nodeid ",child,"'", sep='')
        if(child != node & (!grepl(check, jsonstr, fixed=TRUE))) {
          child_str <- json_prsr(tree_info, child, node_stats)
          jsonstr <- paste(jsonstr, child_str, ',', sep='')
        }
      }
      jsonstr <- substr(jsonstr, 1, nchar(jsonstr)-1) #Remove the comma
      jsonstr <- paste(jsonstr,"]}", sep='')
    }
    return(jsonstr)
  }
