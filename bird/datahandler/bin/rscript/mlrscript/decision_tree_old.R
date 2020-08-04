options(warn=-1)
pathpred <- function(object, ...)
{
  ## coerce to "party" object if necessary
  if(!inherits(object, "party")) object <- as.party(object)
  
  ## get standard predictions (response/prob) and collect in data frame
  rval <- data.frame(response = predict(object, type = "response"))
  rval$prob <- predict(object, type = "prob")
  
  ## get rules for each node
  rls <- partykit:::.list.rules.party(object)
  
  ## get predicted node and select corresponding rule
  rval$rule <- rls[as.character(predict(object, type = "node"))]
  rval=rval[rval$rule==rls[length(rls)[1]],]
  
  return(rval)
}
catcolvalues<-function(catdf,s){
  dep_col <- names(catdf)
  `%ni%` <- Negate(`%in%`)
  catvaluedf=data.frame()
  for (c in dep_col){
    catvaluelist=list(paste(paste(c, catdf[[c]], sep = " is " )))
    newcatvalue=list(paste(paste(c, catdf[[c]], sep = "" ),"",s))
    newcatdf=do.call(rbind, Map(data.frame, catvaluelist=catvaluelist, newcatvalue=newcatvalue))
    catvaluedf=rbind(catvaluedf,newcatdf)}
  return(catvaluedf)
}

#Decision Tree : ( Dependent variable, Independent variable ,method (Anova, Poison, class, exp)
decision_trees<-function(df,y,parameters1="None",features,parameters2="None"){
  df<-subset(df,select=c(features))
  df[,y]=as.factor(df[,y]) 
  depname=y
  train_rows<-sample(x=1:nrow(df),size=0.70*nrow(data))
  train_data<-df[train_rows,]
  test<-df[-train_rows,]
  if(parameters1=="None"){
    trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
    set.seed(3333)
    form=as.formula(paste(y,"~."))
    ptm <- proc.time()
    dtree_fit <- train(form, data = train_data, method = 'rpart',
                       parms = list(split = "information"),
                       trControl=trctrl,
                       tuneLength = 10)
    modeltime=proc.time() - ptm
  }else{
    parameters1_obj<-fromJSON(parameters1);
    print(parameters1_obj)
    set.seed(9876)
    trctrl <- trainControl(method=parameters1_obj$method,number=parameters1_obj$decisiontreenumber, repeats =parameters1_obj$decisiontreerepeats,
                           adaptive = list(min = 2, alpha = 0.05,method = "gls", complete = TRUE))
    prune.control = rpart.control(minsplit = 1)
    form=as.formula(paste(y,"~."))
    parameters2_obj<-fromJSON(parameters2);
    print(parameters2_obj)
    ptm <- proc.time()
    dtree_fit <-train(form, data =train_data , method = "rpart",
                      parms = list(split = parameters2_obj$split),
                      trControl=trctrl,
                      tuneLength = parameters2_obj$decisiontreetunelength,control = prune.control)
    modeltime=proc.time() - ptm
  }
  variable_importance<-data.frame(varImp(dtree_fit)$importance)
  variable_importance<-subset(variable_importance,Overall>=0.001)
  variable_importance$colnames<-rownames(variable_importance)
  variable_importance<-variable_importance[rev(order(variable_importance$Overall)),]
  tree_info<-dtree_fit$finalModel
  print(tree_info)
  tree_info_json = json_prsr(tree_info, node = 1, node_stats = NULL)
  print(tree_info_json)
  prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)
  name_df = setdiff(names(df),y)
  to_predict = data.frame(test[,-which(names(test)==y)])
  names(to_predict)=name_df 
  predicted= predict(dtree_fit,to_predict)
  predict<-as.data.frame(predicted)
  y=as.factor(y)
  predict_predicted=as.factor(predict$predicted)
  test_y = as.factor(as.character(test[,names(test)==y]))
  cm=confusionMatrix(predict_predicted,test_y)
  cm_table<-as.data.frame(cm$table)
  accuracy<-round(cm$overall[1]*10)
  tab<-cm$table
  precision <- data.frame("precision"=(diag(tab) / rowSums(tab)))
  precision$levels<-rownames(precision)
  recall <-data.frame("recall"= (diag(tab) / colSums(tab)))
  recall$levels<-rownames(recall)
  rp_pred <- pathpred(dtree_fit$finalModel)
  freetext<-rp_pred[1,]
  freelanguagetext=paste0("when", " ",freetext$rule, " the prediction of  ", depname," ","is more likely to be"," ",freetext$response  )[1]
  freelanguagetext=freelanguagetext %>% str_c(collapse = "---") %>% str_replace_all(c("maritalmarried < 0.5"="marital status is married","maritalsingle < 0.5"="marital status is single","maritaldivorced < 0.5"="marital status is divorced","maritalmarried > 0.5"="marital status is not married","maritalsingle > 0.5"="marital status is not single","maritaldivorced > 0.5"="marital status is not divorced",">=" = "value is greater than or equal to", ">" = "value is greater than", "&" = "and","<"="value is less than","<="=" value is less than or equal to "))
  cat<-names(select_if(df,is.factor))
  cat_data<-df[ , cat]
  newdtf=catcolvalues(cat_data,s="value is less than ")
  newdf2=catcolvalues(cat_data,s="value is greater than ")
  newdatset=rbind(newdf2,newdtf)
  library(dplyr)
  newdatset=newdatset %>% mutate_all(as.character)
  for(i in range(1,length(newdatset))){
    if(grepl(newdatset[,'newcatvalue'],freelanguagetext)){
      freelanguagetext=gsub(newdatset[i,'newcatvalue'], newdatset[i,'catvaluelist'], freelanguagetext)
    }
  }
  print(freelanguagetext)
  listed<-list(tree_info_json,precision,recall,accuracy,cm_table,variable_importance,freelanguagetext)
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
  prob <- as.data.frame(predict(tree_info, type = "prob"))
  prob=prob %>% add_rownames() 
  prob$rowname=gsub("\\..*","",prob$rowname)
  probs_uniq<-as.data.frame(unique(prob))
  probs_uniq$predicted_node<-colnames(probs_uniq)[apply(probs_uniq,1,which.max)]
  probs_uniq$node_no<-probs_uniq$rowname
  probs_uniq$node_no=gsub("\\X","",probs_uniq$node_no)
  predicted_node_val<-subset(probs_uniq,node_no==node,select = predicted_node)
  if(nrow(predicted_node_val) == 0){
    predicted_node_val =""
  }else{
    predicted_node_val <- gsub(x = predicted_node_val,pattern = "\\.",replacement = " ")
  }
  if(predicted_node_val =="" & final_rule == "" & node == 1){
    predicted_node_val = "Root@@Root= 1"
  }
  if(is.null(node_stats))
    node_stats <- table(tree_info$fitted[1])
  children <- partykit::nodeids(tree_info, node)
  if (length(children) == 1) {
    ct  <- node_stats[as.character(children)]
    jsonstr <- paste("{","\"name\": \"",children,"\",\"value\":\"",predicted_node_val,"\",\"rule\":\"",final_rule,"\"}", sep='')
  } else {
    jsonstr <- paste("{","\"name\": \"", node,"\",\"value\":\"",predicted_node_val,"\",\"rule\":\"",final_rule, "\", \"children\": [", sep='')
    for(child in children){
      check <- paste("{\"name\": \"", child, "\"", sep='')
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