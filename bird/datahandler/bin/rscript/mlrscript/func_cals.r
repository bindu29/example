h2o.init(nthreads=-1,enable_assertions = FALSE)
library(DBI)
con <- dbConnect(clickhouse::clickhouse(), host="192.168.0.143", port=8123L, user="himanshi", password="himanshi",db="himanshi")
data <-dbGetQuery(con, "select `Row ID` as `Row ID`,`Order ID` as `Order ID`,`Order Date` as `Order Date`,`Ship Date` as `Ship Date`,`Ship Mode` as `Ship Mode`,`Customer ID` as `Customer ID`,`Customer Name` as `Customer Name`,`Segment` as `Segment`,`Country` as `Country`,`City` as `City`,`State` as `State`,`Postal Code` as `Postal Code`,`Region` as `Region`,`Product ID` as `Product ID`,`Category` as `Category`,`Sub-Category` as `Sub-Category`,`Product Name` as `Product Name`,`Sales` as `Sales`,`Quantity` as `Quantity`,`Discount` as `Discount`,`Profit` as `Profit` from himanshi.`cibird_1_report_1_2954` LIMIT 10000");
#con <- dbConnect(clickhouse::clickhouse(), host="192.168.0.152", port=8123L, user="manogna", password="manogna",db="manogna")
#data <-dbGetQuery(con, "select `Row ID` as `Row ID`,`Order ID` as `Order ID`,`Order Date` as `Order Date`,`Ship Date` as `Ship Date`,`Ship Mode` as `Ship Mode`,`Customer ID` as `Customer ID`,`Customer Name` as `Customer Name`,`Segment` as `Segment`,`Country` as `Country`,`City` as `City`,`State` as `State`,`Postal Code` as `Postal Code`,`Region` as `Region`,`Product ID` as `Product ID`,`Category` as `Category`,`Sub-Category` as `Sub-Category`,`Product Name` as `Product Name`,`Sales` as `Sales`,`Quantity` as `Quantity`,`Discount` as `Discount`,`Profit` as `Profit` from manogna.`cibird_1_supersalesreport_1_2723` LIMIT 10000");
#data <-dbGetQuery(con, "select `Month` as `Month`,`Australia` as `Australia`,`Cook Islands` as `Cook Islands`,`Fiji` as `Fiji`,`Samoa` as `Samoa`,`ChinaPRo` as `ChinaPRo`,`India` as `India`,`Thailand` as `Thailand`,`UK` as `UK`,`USA` as `USA`,`Total` as `Total` from manogna.`cibird_1_nzdestnztsreport_1_2724` LIMIT 10000");
data<-as.data.frame(data)

columnsArray<-'[{"columnDisplayName":"Row ID","tableDisplayType":"number","columnName":"Row ID"},{"columnDisplayName":"Order ID","tableDisplayType":"string","columnName":"Order ID"},{"columnDisplayName":"Order Date","tableDisplayType":"date","columnName":"Order Date"},{"columnDisplayName":"Ship Date","tableDisplayType":"date","columnName":"Ship Date"},{"columnDisplayName":"Ship Mode  123","tableDisplayType":"string","columnName":"Ship Mode"},{"columnDisplayName":"Customer ID 123","tableDisplayType":"string","columnName":"Customer ID"},{"columnDisplayName":"Customer Name","tableDisplayType":"string","columnName":"Customer Name"},{"columnDisplayName":"Segment","tableDisplayType":"string","columnName":"Segment"},{"columnDisplayName":"Country 123","tableDisplayType":"string","columnName":"Country"},{"columnDisplayName":"City 123","tableDisplayType":"string","columnName":"City"},{"columnDisplayName":"State","tableDisplayType":"string","columnName":"State"},{"columnDisplayName":"Postal Code","tableDisplayType":"number","columnName":"Postal Code"},{"columnDisplayName":"Region 123","tableDisplayType":"string","columnName":"Region"},{"columnDisplayName":"Product ID","tableDisplayType":"string","columnName":"Product ID"},{"columnDisplayName":"Category 123","tableDisplayType":"string","columnName":"Category"},{"columnDisplayName":"Sub-Category 123","tableDisplayType":"string","columnName":"Sub-Category"},{"columnDisplayName":"Product Name","tableDisplayType":"string","columnName":"Product Name"},{"columnDisplayName":"Sales 123","tableDisplayType":"number","columnName":"Sales"},{"columnDisplayName":"Quantity","tableDisplayType":"number","columnName":"Quantity"},{"columnDisplayName":"Discount","tableDisplayType":"number","columnName":"Discount"},{"columnDisplayName":"Profit 123","tableDisplayType":"number","columnName":"Profit"}]'
columnsArray<-'[{"tableDisplayType":"number","columnName":"Row ID"},{"tableDisplayType":"string","columnName":"Order ID"},{"tableDisplayType":"date","columnName":"Order Date"},{"tableDisplayType":"date","columnName":"Ship Date"},{"tableDisplayType":"string","columnName":"Ship Mode"},{"tableDisplayType":"string","columnName":"Customer ID"},{"tableDisplayType":"string","columnName":"Customer Name"},{"tableDisplayType":"string","columnName":"Segment"},{"tableDisplayType":"string","columnName":"Country"},{"tableDisplayType":"string","columnName":"City"},{"tableDisplayType":"string","columnName":"State"},{"tableDisplayType":"number","columnName":"Postal Code"},{"tableDisplayType":"string","columnName":"Region"},{"tableDisplayType":"string","columnName":"Product ID"},{"tableDisplayType":"string","columnName":"Category"},{"tableDisplayType":"string","columnName":"Sub-Category"},{"tableDisplayType":"string","columnName":"Product Name"},{"tableDisplayType":"number","columnName":"Sales"},{"tableDisplayType":"number","columnName":"Quantity"},{"tableDisplayType":"number","columnName":"Discount"},{"tableDisplayType":"number","columnName":"Profit"}]'
columnsArray <- fromJSON(columnsArray)

## FOR Correlation & Chi square#####
features<-c(names(data));
changed_data <- columns_data_type(data,columnsArray)
correlationData<-correlations(changed_data)
if(data.frame(correlationData[5]) == "no categorical data"){
  chisquare_Data<-"no categorical data"
}else{
  chisquare_Data<-ch_sq_test(data.frame(correlationData[5]))
}
if(length(data.frame(correlationData[2]))>0){
  cor_Data<-correlationData[2];
}else{
  cor_Data<-"no numerical data";
}
result<-list(cor_Data,chisquare_Data)
dbDisconnect(con);
remove(con);
#return(result);

###### FOR linear regression#####
con <- dbConnect(clickhouse::clickhouse(), host="192.168.0.152", port=8123L, user="himanshi", password="himanshi",db="himanshi")
data <-dbGetQuery(con, "select `Date` as `Date`,`Open` as `Open`,`High` as `High`,`Low` as `Low`,`Close` as `Close`,`WAP` as `WAP`,`No of Shares` as `No of Shares`,`No of Trades` as `No of Trades`,`Total Turnover` as `Total Turnover`,`Deliverable Quantity` as `Deliverable Quantity`,`Deli  Qty to Traded Qty` as `Deli  Qty to Traded Qty`,`Spread H L` as `Spread H L`,`Spread C O` as `Spread C O` from himanshi.`cibird_1_goldpricereport_1_2825` LIMIT 10000");
data<-as.data.frame(data)
y="Sales"
changed_data <- columns_data_type(data,columnsArray)
correlationData<-correlations(changed_data)
complete_data<-correlationData[3]
new_complete_data<-complete_data[[1]]
features<-c(names(new_complete_data));

new<-transformation(new_complete_data,y)

comb_data<-new_complete_data
# an.error.occured <- FALSE
# tryCatch( { vars<-variable_importance_h2o(new,y);}
#           , error = function(e) {an.error.msg<<-"Variable importance cannot be computed with single independent variable"
#           an.error.occured = TRUE
#           })

vars<-variable_importance_h2o(new,y)
if(class(vars)=="data.frame"){
  important_vars<-vars$x
  subset_data <- new[,which(colnames(new)%in%important_vars)]
  subset_data[,y]<-new[,y]
  tarname<-colnames(subset_data)[which(names(subset_data) == "target_var")] <- y 
}else{
  subset_data<-new
}


parametersObj<-'{"nfolds":5,"family":"gaussian","balance_classes":"FALSE"}'
parameters ="NONE"  
if(parametersObj!="NONE"){
  parameters = parametersObj;
}

features<-names(subset_data)
model_data<-model(subset_data,tarname)
subset_data1<-subset_data[,-which(names(subset_data)==y)]
simulation_var<-data_str(subset_data1)
model_coff<-data.frame(model_data[4])
test<-subset_data[sample(nrow(subset_data), 1), ]
test<-test[,-which(names(test)==y)]
final_simulation<-simulation(test,model_coff)
dfs<-modelling(subset_data,y,parameters,vars,features = features,final_simulation,model_coff,test,columnsArray)

df<-subset_data


da<-data.frame(dfs[2])
names(da)


## K Means


#excludedata<-c("State","Profit")
#excludedata <- gsub(x = excludedata,pattern = "<\\ >",replacement = "_")
#y<-gsub(x=y,pattern = "<\\ >",replacement = "_")
#allvars <- names(data) %in% excludedata
#data_after_excluding <- data[!allvars]

changed_data <- columns_data_type(data,columnsArray)
correlationData<-correlations(changed_data)
complete_data<-as.data.frame(correlationData[3])
#new_complete_data<-complete_data[[1]]
new<-transformation(complete_data)
parameters="None"
parametersObj="{\"max_iterations\":5,\"score_each_iteration\":\"False\",\"ignore_const_cols\":\"True\",\"kvalue\":5,\"max_runtime_secs\":10,\"categoricalencoding\":\"AUTO\",\"standardize\":\"False\",\"estimate_k\":\"True\"}"

if(fromJSON(parametersObj)!="None"){
  parameters = parametersObj
}
df = new
dfs<-kmeansmodel(new,parameters) 
data =data.frame(dfs[1])
cluster_profile_data<-cluster_profiling(data.frame(dfs[1]))
#print(cluster_profile_data)
dfs[8]<-cluster_profile_data[2];
nlgdata1<-cluster_nlg(withiness_tab,betweeness)
nlgdata2<-profiling_nlg(mean_num,frq)
dfs[11]<-cluster_nlg(data.frame(dfs[4]),data.frame(dfs[6]))
dfs[12]<-profiling_nlg(data.frame(cluster_profile_data[4]),data.frame(cluster_profile_data[5]))


##### Decision Tree###############
h2o.init()
h2o.init(nthreads=-1,enable_assertions = FALSE) 
library(DBI)
con <- dbConnect(clickhouse::clickhouse(), host="192.168.0.152", port=8123L, user="himanshi", password="himanshi",db="himanshi")
data <-dbGetQuery(con, "select `Row ID` as `Row ID`,`Order ID` as `Order ID`,`Order Date` as `Order Date`,`Ship Date` as `Ship Date`,`Ship Mode` as `Ship Mode`,`Customer ID` as `Customer ID`,`Customer Name` as `Customer Name`,`Segment` as `Segment`,`Country` as `Country`,`City` as `City`,`State` as `State`,`Postal Code` as `Postal Code`,`Region` as `Region`,`Product ID` as `Product ID`,`Category` as `Category`,`Sub-Category` as `Sub-Category`,`Product Name` as `Product Name`,`Sales` as `Sales`,`Quantity` as `Quantity`,`Discount` as `Discount`,`Profit` as `Profit` from himanshi.`cibird_1_report_1_2909` LIMIT 10000");
data<-as.data.frame(data)
columnsArray<-'[{"tableDisplayType":"number","columnName":"Row ID"},{"tableDisplayType":"string","columnName":"Order ID"},{"tableDisplayType":"date","columnName":"Order Date"},{"tableDisplayType":"date","columnName":"Ship Date"},{"tableDisplayType":"string","columnName":"Ship Mode"},{"tableDisplayType":"string","columnName":"Customer ID"},{"tableDisplayType":"string","columnName":"Customer Name"},{"tableDisplayType":"string","columnName":"Segment"},{"tableDisplayType":"string","columnName":"Country"},{"tableDisplayType":"string","columnName":"City"},{"tableDisplayType":"string","columnName":"State"},{"tableDisplayType":"number","columnName":"Postal Code"},{"tableDisplayType":"string","columnName":"Region"},{"tableDisplayType":"string","columnName":"Product ID"},{"tableDisplayType":"string","columnName":"Category"},{"tableDisplayType":"string","columnName":"Sub-Category"},{"tableDisplayType":"string","columnName":"Product Name"},{"tableDisplayType":"number","columnName":"Sales"},{"tableDisplayType":"number","columnName":"Quantity"},{"tableDisplayType":"number","columnName":"Discount"},{"tableDisplayType":"number","columnName":"Profit"}]'
columnsArray <- fromJSON(columnsArray)

lapply(list('partykit','rattle','rpart.plot','rpart','caret','jsonlite','devtools','ps','aCRM','DBI','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx','MLmetrics'), require, character.only = TRUE)

y="marital"
y="Ship Mode"
y<-gsub(x=y,pattern = "\\ ",replacement = "_")
y<-gsub("[^[:alnum:]///' ]", "", y)
changed_data <- columns_data_type(data,columnsArray)
correlationData<-correlations(changed_data)
complete_data<-correlationData[3]
new_complete_data<-complete_data[[1]]
#features<-c(names(new_complete_data));
#features <- gsub(x = features,pattern = "\\ ",replacement = "_")
parameters1<-'{"method":"repeatedcv","decisiontreenumber":5,"decisiontreerepeats":3}'
parameters2<-'{"split":"information","decisiontreetunelength":10}'
features<-names(new_complete_data)
features <- gsub(x = features,pattern = "\\ ",replacement = "_")
features<-gsub("[^[:alnum:]///' ]", "", features)
names(new_complete_data)<-features
df = new_complete_data
listed<-decision_trees(new_complete_data,y,parameters1,features,parameters2)
fancyRpartPlot(dtree_fit1$finalModel)

#co-occurance
new<-data[,c(2,3)]
a=co_occurrence(data$Category)
b<-list()

textColumns ="[\"Review Text\"]"
text=c(fromJSON(textColumns))
#text<-gsub(x=text,pattern = "\\ ",replacement = "_")
data<-na.omit(data)
df<-subset(data,select=c(text))
finaldata<-lapply(df, co_occurrence)


##########forecast
#Choose a variable to forecast

Variable = "Sales"
#Date = "Ship Date"
Date = c("Order Date")
Model = "NNETAR"
no_of_periods_to_forecast = 12
if(Model == "NNETAR"){
  IndependentVariables = c("Profit","Discount")
  future_values =c(10,12)
}else{
  IndependentVariables = "None"
}
data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)],as.factor)
data = knnImputation(data)
data$`Order Date` = as.Date(data$`Order Date`)
data[,Date] = as.Date(data[,Date])
result<-forecasting_model(data,Variable,Date,Model,no_of_periods_to_forecast,IndependentVariables,future_values)

#as.yearmon(as.Date(Data_forecasting$Date))
Variable = "Total"
Date = "Month"
Model = "NNETAR"
no_of_periods_to_forecast = 12
if(Model == "NNETAR"){
  IndependentVariables = c("USA","India")
  future_values =c(10,12)
}else{
  IndependentVariables = "None"
}
Data_initial = data


##########Logistic Regression

h2o.init()
h2o.init(nthreads=-1,enable_assertions = FALSE) 
library(DBI)
con <- dbConnect(clickhouse::clickhouse(), host="192.168.0.152", port=8123L, user="himanshi", password="himanshi",db="himanshi")
data <-dbGetQuery(con, "select `AGE` as `AGE`,`WORKCLASS` as `WORKCLASS`,`FNLWGT` as `FNLWGT`,`EDUCATION` as `EDUCATION`,`EDUCATIONNUM` as `EDUCATIONNUM`,`MARITALSTATUS` as `MARITALSTATUS`,`OCCUPATION` as `OCCUPATION`,`RELATIONSHIP` as `RELATIONSHIP`,`RACE` as `RACE`,`SEX` as `SEX`,`CAPITALGAIN` as `CAPITALGAIN`,`CAPITALLOSS` as `CAPITALLOSS`,`HOURSPERWEEK` as `HOURSPERWEEK`,`NATIVECOUNTRY` as `NATIVECOUNTRY`,`ABOVE50K` as `ABOVE50K` from himanshi.`cibird_1_adultreport_1_2833` LIMIT 10000");
data<-as.data.frame(data)

lapply(list('partykit','rattle','rpart.plot','rpart','caret','jsonlite','devtools','ps','aCRM','DBI','dataPreparation', 'dplyr','e1071', 'mice','h2o','plyr','data.table','dummies','stringr','tidyverse','caret','taRifx','MLmetrics'), require, character.only = TRUE)

y<-"ABOVE50K"
y<-gsub(x=y,pattern = "\\ ",replacement = "_")
y<-gsub("[^[:alnum:]///' ]", "", y)
#features<-c("AGE","WORKCLASS","EDUCATIONNUM","MARITALSTATUS","RELATIONSHIP")
changed_data <- columns_data_type(data,columnsArray)
correlationData<-correlations(changed_data)
complete_data<-correlationData[3]
new_complete_data<-complete_data[[1]]
features<-names(new_complete_data)
features <- gsub(x = features,pattern = "\\ ",replacement = "_")
features<-gsub("[^[:alnum:]///' ]", "", features)
names(new_complete_data)<-features
new<-transformation(new_complete_data,features,y)
vars<-variable_importance_h2o(new,y)
df = new_complete_data
parameters="None"
listed<-modelling_GBM(df,y,parameters="None",features,vars)