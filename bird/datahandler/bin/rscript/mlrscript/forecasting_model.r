forecasting_model<-function(Data_initial,Variable,Date,Model,no_of_periods_to_forecast,IndependentVariables,future_values,yDisplay){ 
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
    
    slope_text = paste0("<li class=\"nlg-item\"><span class=\"nlgtext\"> The slope of time for <font style=\"color: #0089ff;\">",yDisplay," </font> is <b>",round(slope_with_time,4),"</b> , For every one unit change in time <font style=\"color: #0089ff;\">",yDisplay," </font> is effected by <b>",round(slope_with_time,4) ,"</b> units </span></li>")
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
  
  seasonal_text = paste0("The amount of ", yDisplay," is effected due to seasonality with time")
  
  seasonal_component = data.frame((head(decomp_overall[,c("month","seasonality")],12)))
  names(seasonal_component)<-c("Month","Seasonality")
  
  model_text = paste0("Model has Forecasted the ",yDisplay ," for next ",no_of_periods_to_forecast ," months")
  #table forecast
  forecasted_fin = na.omit(forecasted[,c("Date","Point.Forecast")])
  names(forecasted_fin)<-c("Date",paste0(yDisplay," Forecast values"))
  Percentage_Variance_Explained_by_trend_text = paste0("<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #0089ff;\"> Trend </font> is <b>",round(Percentage_Variance_Explained_by_trend,4),"</b> </span></li>")
  Percentage_Variance_Explained_by_seasonality_text = paste0("<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #0089ff;\"> Seasonality </font>is <b>",round(Percentage_Variance_Explained_by_seasonality,4),"</b> </span></li>")
  Percentage_Variance_Explained_by_randomness_text  = paste0("<li class=\"nlg-item\"><span class=\"nlgtext\"> Variance Explained By <font style=\"color: #0089ff;\"> Randomness </font> is <b>",round(Percentage_Variance_Explained_by_randomness,4),"</b> </span></li>")
  
  listed<-list(decomp_overall,slope_text,Percentage_Variance_Explained_by_trend_text,Percentage_Variance_Explained_by_seasonality_text,
               Percentage_Variance_Explained_by_randomness_text,seasonal_text,seasonal_component,model_text,forecasted,forecasted_fin,MAPE,MSE,ME,MAE)
  return(listed)  
}