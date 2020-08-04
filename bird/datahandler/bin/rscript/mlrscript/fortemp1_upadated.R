library(lubridate)
library(zoo)
library(xts)
library(dplyr)
library(highcharter)
library(forecast)
library(gtools)
library(stats)

Data_initial = read.csv("C:/Users/Viswas/Downloads/ML stuff/data sets/New folder/supersales.csv")

#Assuming the data is already cleaned in previous steps (missing values imputed and date is in date format)
#the below commented steps are not required based on above assumption

  library(DMwR)
  #data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)],as.factor)
  #Data_initial = data 
  Data_initial = knnImputation(Data_initial)
  #Data_initial$`Month` = as.Date(Data_initial$`Month`)
  #Data_initial$`Order Date` = as.Date(Data_initial$`Order Date`)
  Data_initial$`Order Date` = as.Date(Data_initial$`Order Date`,format = "%d-%m-%Y")

  
Variable = "Sales"
Date = "Order Date"
Model = "NNETAR"
no_of_periods_to_forecast = 12
if(Model == "NNETAR"){
  IndependentVariables = c("Profit","Discount")
  future_values =c(10,12)
}else{
  IndependentVariables = "None"
}


############################################### INPUTS #####################################
#____________________________________________________________________________________________
# #Choose a variable to forecast
# Variable = "Monthly_sales_total"
# Date = "Date"
# Model = "NNETAR"
# no_of_periods_to_forecast = 10
# if(Model == "NNETAR"){
#   IndependentVariables = c("Price_Per_Unit","Discount_Percentage")
#   future_values =c(10,12)
# }else{
#   IndependentVariables = "None"
# }


############################################# PROCESS and Model building
names(Data_initial)
Data_forecasting = Data_initial[,c(Date,Variable)]

names(Data_forecasting) = c("Date","Value")
if(IndependentVariables != "None"){
  Data_forecasting = cbind(Data_forecasting,Data_initial[,c(IndependentVariables)])
}else{
  Data_forecasting = Data_forecasting
}
class(Data_forecasting$Date)

days_available = length(unique(as.numeric(days((Data_forecasting$Date)))))
months_available = length(unique(as.numeric(as.yearmon((Data_forecasting$Date)))))
years_available =  length(unique(as.numeric(format(Data_forecasting$Date,'%Y'))))
aggregation_level = c()
if((days_available/months_available) > 25){
  aggregation_level = append(aggregation_level,"Daily")
}
if((months_available/years_available) >= 10){
  aggregation_level = append(aggregation_level,"Monthly")
}
if((years_available) >= 24){
  aggregation_level = append(aggregation_level,"Yearly")
}

if(length(aggregation_level) == 0){
  aggregation_level = "Error"
  Error = paste0("Insufficient data for time series analysis with ",days_available," Days, ",months_available," Months and ",years_available," Years")
  #print(Error)
}else{
  Error = "none"
  print(aggregation_level)
}

if(aggregation_level == "Error"){
  print(Error)
}else{
  #the aggregation can be changed to dynamic based on user selected options from dynamic list of options from aggregation level
  if(IndependentVariables == "None"){
    monthly_aggregated = aggregate(Value~as.yearmon(Date),Data_forecasting,FUN = sum)
    names(monthly_aggregated) = c("Date","Value")
      }else{
    monthly_aggregated = aggregate(.~as.yearmon(Date),Data_forecasting,FUN = sum)
    monthly_aggregated = monthly_aggregated[,-c(which(colnames(monthly_aggregated)=="Date"))]
    colnames(monthly_aggregated)[1] = "Date"
          }  

  monthly_xts = xts(monthly_aggregated$Value,order.by = monthly_aggregated$Date,frequency = 12)
  #monthly_xts$frequency <-12
  attr(monthly_xts, 'frequency') <- 12  # Set the frequency of the xts object
  plot(monthly_xts)
  v = tryCatch({
    #decomposition
    decomposition = stats::decompose(as.ts(monthly_xts))
    total_overall = data.frame(round(decomposition$x))
    trend_overall = data.frame(round(decomposition$trend))
    seasonality_overall = data.frame(round(decomposition$seasonal,2))
    randomness_overall = data.frame(round(decomposition$random))
    decomp_overall = cbind(total_overall,trend_overall,seasonality_overall,randomness_overall)
    decomp_overall$Date = monthly_aggregated$Date
    names(decomp_overall) = c("Actual","trend","seasonality","randomness","Date")
    decomp_overall <- transform(decomp_overall, Date = as.Date(Date, frac = 1))
    Error = "none"
    decomp_overall$month = month(decomp_overall$Date)
    decomp_overall
    print(decomp_overall)
    
  },error = function(e){
    Error = "Unable to Decompose time series"
    print(Error)
  })
}

if(Error != "none" ){
  Error = Error
}else{
  decomp_overall = v
  Percentage_Variance_Explained_by_trend = (var(na.omit(decomp_overall$trend))/var(na.omit(decomp_overall$Actual)))*100
  Percentage_Variance_Explained_by_seasonality = (var(na.omit(decomp_overall$seasonality))/var(na.omit(decomp_overall$Actual)))*100
  Percentage_Variance_Explained_by_randomness = (var(na.omit(decomp_overall$randomness))/var(na.omit(decomp_overall$Actual)))*100
  
  slope_with_time = lm(decomp_overall$Actual~c(1:nrow(decomp_overall)))
  slope_with_time = as.numeric(slope_with_time$coefficients[2])
  
  slope_text = paste0("The slope of time for ",Variable," is ",slope_with_time," , For every one unit change in time ",Variable," is effected by ",slope_with_time ," units")
  seasonal_table = as.data.frame(decomp_overall$seasonality)

    
    
if(Model == "Holtwinters"){
  #model building
  model_monthly = HoltWinters(monthly_xts)
  fitted = as.data.frame(model_monthly$fitted)
  model_fin = tail(monthly_aggregated,-12)
  model_fin$fitted = fitted$xhat
  forecasted = as.data.frame(forecast(model_monthly,h = no_of_periods_to_forecast))
  
}else{
  if(Model == "ARIMA"){
    model_monthly = auto.arima(monthly_xts)
    fitted = as.data.frame(model_monthly$fitted)
    model_fin = monthly_aggregated
    model_fin$fitted = fitted$x
    forecasted = as.data.frame(forecast(model_monthly,h = no_of_periods_to_forecast))
    
  }else{
    if(Model == "NNETAR"){
      model_monthly = nnetar(monthly_xts,xreg = monthly_aggregated[,IndependentVariables] )
      fitted = as.data.frame(model_monthly$fitted)
      model_fin = monthly_aggregated
      model_fin$fitted = fitted$x
      xreg_data = data.frame(t(data.frame(IndependentVariables,future_values)))
      colnames(xreg_data) <- as.character(unlist(xreg_data[1,]))
      xreg_data = xreg_data[-1,]  
      rownames(xreg_data) = NULL
      xreg_data = xreg_data[rep(seq_len(nrow(xreg_data)), each=no_of_periods_to_forecast),]
      xreg_data = data.frame(apply(xreg_data,2,function(x){as.numeric(x)}))
      forecasted = forecast(model_monthly,h = no_of_periods_to_forecast,xreg = xreg_data)
      forecasted = data.frame(t(data.frame(forecasted)))
      forecasted = data.frame(apply(forecasted,2,function(x){as.numeric(as.character(x))}))
      Forecasted = data.frame()
      for(i in 1: ncol(forecasted)){
        forecasted_l = na.omit(forecasted[i])
        names(forecasted_l) = "Point.Forecast"
        Forecasted = na.omit(rbind(Forecasted,forecasted_l))
      }
      forecasted = Forecasted
    }
  }
}
model_monthly
model_fin = na.omit(model_fin)
MAPE = mean((abs(model_fin$Value - model_fin$fitted)/abs(model_fin$Value))*100)
MSE = mean((model_fin$Value - model_fin$fitted)^2)
ME = mean(model_fin$Value - model_fin$fitted)
MAE = mean(abs(model_fin$Value - model_fin$fitted))

forecasted$Date=seq(max(model_fin$Date)+(1/12),max(model_fin$Date)+(1/12)*no_of_periods_to_forecast,by = (1/12))
forecasted = merge(model_fin,forecasted,by = "Date",all = T,no.dups = T )
forecasted <- transform(forecasted, Date = as.Date(Date, frac = 1))
}






print(Error)


if(Error == "none"){
  ###############################output one 
  #plot Actual with Date from decomp_overall data
  #decomp_overall
  hc_overall <- highchart(type = "stock") %>%
    hc_add_series_times_values(decomp_overall$Date,decomp_overall$Actual,color = "#0000FF", name = "Actual")%>%
    hc_rangeSelector(inputEnabled = T)  %>%
    hc_xAxis(labels = T)%>%
    hc_scrollbar(enabled = T)%>%
    hc_legend(enabled = T)
  hc_overall
  
  #################################outputtwo
  
  slope_text
  
  
  #################################output three (all the three plots trend seasonality and randomness)
  
  #decomp_overall is the data
  
  #trend and date
  hc_trend <- highchart(type = "stock") %>%
    hc_add_series_times_values(decomp_overall$Date,decomp_overall$trend,color = "#228B22", name = "Trend")%>%
    hc_rangeSelector(inputEnabled = T)  %>%
    hc_xAxis(labels = T)%>%
    hc_scrollbar(enabled = T)%>%
    hc_legend(enabled = T)
  
  #seasonality and date
  hc_seasonality <- highchart(type = "stock") %>%
    hc_add_series_times_values(decomp_overall$Date,decomp_overall$seasonality,color = "#FF7F50", name = "Seasonality")%>%
    hc_rangeSelector(inputEnabled = T)  %>%
    hc_xAxis(labels = T)%>%
    hc_scrollbar(enabled = T)%>%
    hc_legend(enabled = T)
  
  #randomness and date
  hc_randomess <- highchart(type = "stock") %>%
    hc_add_series_times_values(decomp_overall$Date,decomp_overall$randomness,color = "#000000", name = "Randomness")%>%
    hc_rangeSelector(inputEnabled = T)  %>%
    hc_xAxis(labels = T)%>%
    hc_scrollbar(enabled = T)%>%
    hc_legend(enabled = T)
  
  
  ############output 4 (all the three variance boxes)
  Percentage_Variance_Explained_by_trend
  Percentage_Variance_Explained_by_seasonality
  Percentage_Variance_Explained_by_randomness
  
  #####################output 5 (seasonal component)
  seasonal_text = paste0("The amount of ", Variable," is effected due to seasonality with time")
  
  seasonal_component = data.frame(t(head(decomp_overall[,c("month","seasonality")],12)))
  
  ####################output 6 model outputs
  model_monthly #model output
  
  model_text = paste0("Model has forecasted the ",Variable ," for next ",no_of_periods_to_forecast ," months") 
  
  #plot forecast
  if(Model == "NNETAR"){
    hc_forecasted_overall <- highchart(type = "stock") %>% 
      hc_add_series_times_values(forecasted$Date,forecasted$Value,color = "#006400",name = "Actual")%>% 
      hc_add_series_times_values(forecasted$Date,forecasted$fitted,color = "#696969",name = "Fitted")%>% 
      hc_add_series_times_values(forecasted$Date,forecasted$Point.Forecast,color = "#FFA500",name = "Forecast")%>% 
      hc_rangeSelector(inputEnabled = T)  %>% 
      hc_scrollbar(enabled = T) %>%
      hc_legend(enabled = T)
  }else{
    hc_forecasted_overall <- highchart(type = "stock") %>% 
      hc_add_series_times_values(forecasted$Date,forecasted$Value,color = "#006400",name = "Actual")%>% 
      hc_add_series_times_values(forecasted$Date,forecasted$fitted,color = "#696969",name = "Fitted")%>% 
      hc_add_series_times_values(forecasted$Date,forecasted$Point.Forecast,color = "#FFA500",name = "Forecast")%>% 
      hc_add_series_times_values(forecasted$Date,forecasted$Lo.95,color = "#0000A0",name = "Lower 95")%>% 
      hc_add_series_times_values(forecasted$Date,forecasted$Hi.95,color = "#0000A0",name = "Higher 95")%>% 
      hc_rangeSelector(inputEnabled = T)  %>% 
      hc_scrollbar(enabled = T) %>%
      hc_legend(enabled = T)
  }
  
  #table forecast
  forecasted_fin = na.omit(forecasted[,c("Date","Point.Forecast")])
  hc_forecasted_overall
  
  #metrics
  MAPE
  MSE
  ME
  MAE 
}else{
  print(Error)
}




























