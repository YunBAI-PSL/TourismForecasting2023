mase_cal <- function(insample, outsample, forecasts) {
  stopifnot(stats::is.ts(insample))
  #Used to estimate MASE
  frq <- stats::frequency(insample)
  forecastsNaiveSD <- rep(NA,frq)
  for (j in (frq+1):length(insample)){
    forecastsNaiveSD <- c(forecastsNaiveSD, insample[j-frq])
  }
  masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
  
  outsample <- as.numeric(outsample) ; forecasts <- as.numeric(forecasts)
  mase <- (abs(outsample-forecasts))/masep
  return(mase)
}

# winkler_cal<-function(outsample,forecasts.lower,forecasts.upper,a=0.05){
#   w=c()
#   for (t in 1:length(outsample)) {
#     w_t=forecasts.upper[t]-forecasts.lower[t]
#     if (forecasts.lower[t]>outsample[t]){
#       w=c(w,w_t+2*(forecasts.lower[t]-outsample[t])/a)
#     }else if(forecasts.upper[t]<outsample[t]){
#       w=c(w,w_t+2*(outsample[t]-forecasts.upper[t])/a)
#     }else{
#       w=c(w,w_t)
#     }
#   }
#   return(w)
# }


#get start and end dates for each time series
get.start.end.dates<-function(ts,dates){
  index=which(!is.na(ts)) 
  len.ts=length(index)
  start.date=dates[index[1]]
  end.date=dates[index[len.ts]]
  c(start.date,end.date,len.ts,index[1],index[len.ts])
}



#extract month and year
extract.y.m<-function(date.char){
  year=as.integer(substr(date.char,1,4))
  month=as.integer(substr(date.char,6,7))
  c(year,month)
}



#ts.data: vector including missing data 
#average: forecasts and snaive
fill.missing<-function(ts.data,dates,averge.option){
  ts.data.copy=ts.data
  # c(start.date,end.date,len.ts,index[1],index[len.ts])
  r=get.start.end.dates(ts.data,dates)
  print('res')
  print(r)
  #extract start/end dates: y/m
  y.m.start=extract.y.m(r[1])
  y.m.end=extract.y.m(r[2])
  #extract end dates: y/m
  year.end=y.m.end[1]
  month.end=y.m.end[2]+1
  if (month.end>12){
    month.end=month.end-12
  }
  print('end.month.non.missing')
  print(month.end)
  #month.end=7
  #find corresponding observations in 2019 to impute these missing data
  index1=which(dates == paste('2019M0',toString(month.end),sep = ''))
  index2=which(dates == '2019M07')
  indexes=which(!is.na(ts.data)) 
  len.ts=length(indexes)
  print('len.ts')
  print(len.ts)
  print(length(ts.data))
  print('index1-2')
  print(index1)
  print(index2)
  #num.of.missing.data=index2-index1+1
  #missing data filling using the observations of same seasons in 2019 after covid
  ts.data[(length(ts.data)-index2+index1):length(ts.data)]=ts.data[index1:index2]
  #missing data filling before covid
  
  ts.object=na.interp(ts(ts.data[r[4]:length(ts.data)],start=c(y.m.start[1],y.m.start[2]),frequency=12))
  print('ts1')
  print(ts.object)
  if (averge.option){
    #ts.object1=na.interp(ts(ts.data.copy[r[4]:length(ts.data)],start=c(y.m.start[1],y.m.start[2]),frequency=12))
    #make predictions
    ts.object1=na.interp(ts(ts.data.copy[r[4]:r[5]],start=c(y.m.start[1],y.m.start[2]),frequency=12))
    #model=auto.arima(ts.object1)
    model=baggedModel(ts.object1)
    #model=snaive(ts.object1)
    preds=forecast(model,h=(index2-index1+1))
    #fill missing data using preds
    ts.object1=ts(c(ts.object1,preds$mean),start=c(y.m.start[1],y.m.start[2]),frequency=12)
    ts.object=(ts.object+ts.object1)/2
    print('ts2')
    print(ts.object1)
  }
  return(ts.object)
}



#test
#handle missing data using snavive, and then fit model
forecast.one.year.missing.proc<-function(data,train.end.date,h,method.option,averge.option){
  library(gridExtra)
  library(ggplot2)
  library(forecast)
  dates=as.vector(unlist(data[,1]))
  index=which(dates==train.end.date)
  p <- list()
  coln=colnames(data)
  #accuracy.df=data.frame(country=coln[2:length(coln)],mase=rep(0,20))
  point.preds.df=c()
  lower.preds.df=c()
  upper.preds.df=c()
  # point.preds.df= data.frame(matrix(0,nrow = h, ncol = 20)) 
  # lower.preds.df= data.frame(matrix(0,nrow = h, ncol = 20)) 
  # upper.preds.df= data.frame(matrix(0,nrow = h, ncol = 20)) 
  ts.list=vector(mode='list', length=20)
  for (i in 2:length(coln)) {
    # print('i')
    # print(i)
    print(coln[i])
    ts.data=as.vector(unlist(data[,i]))
    #c(start.date,end.date,len.ts,index[1],index[len.ts])
    #r=get.start.end.dates(ts.data,dates)
    #extract start dates: y/m
    # y.m.start=extract.y.m(r[1])
    # y.m.end=extract.y.m(r[2])
    # 
    #ts.object: train+test
    #handle missing data
    
    #ts.object=na.interp(ts(ts.data[r[4]:length(ts.data)],start=c(y.m.start[1],y.m.start[2]),frequency=12))
    #missing data processing
    ts.object=fill.missing(ts.data,dates,averge.option)
    #print(ts.object)
    ts.list[[i-1]]=ts.object
    
    #x: train
    #xx:test
    ts.entry <- list(x =ts.object, 
                     n = (length(ts.object)), 
                     h = h)
    if(method.option=='arima'){
      model=auto.arima(ts.entry$x)
    }else if(method.option=='ets'){
      model=ets(ts.entry$x)
    } else if (method.option=='snaive'){
      model=snaive(ts.entry$x)
    } else if (method.option=='bagging'){
      print('test')
      model=baggedModel(ts.entry$x)
      lower.p=rep(0,h)
      upper.p=rep(0,h)
      num=length(model$models)
      for (i in 1:num) {
        res=forecast(model$models[[i]],h=h)
        lower.p=lower.p+c(res$lower[,1])
        upper.p=upper.p+c(res$upper[,1])
      }
      lower.p=lower.p/num
      upper.p=upper.p/num
    }
    preds=forecast(model,h=ts.entry$h)
    #plot forecasts
    p[[i]]<- autoplot(ts.object) +
      autolayer(preds$mean, series="Mean") +
      xlab("Year") + ylab("Visits") +
      ggtitle(coln[i]) +
      guides(colour=guide_legend(title="Forecast"))
    
    #plot(preds)
    if(method.option=='bagging'){
      print('lower.p')
      print(lower.p)
      print("upper.p")
      print(upper.p)
      print(lower.preds.df)
      print(upper.preds.df)
      print('mean')
      print(c(preds$mean))
      point.preds.df=cbind(point.preds.df,c(preds$mean))
      lower.preds.df=cbind(lower.preds.df,lower.p)
      upper.preds.df=cbind(upper.preds.df,upper.p)
    }else{
      point.preds.df=cbind(point.preds.df,c(preds$mean))
      lower.preds.df=cbind(lower.preds.df,c(preds$lower[,1]))
      upper.preds.df=cbind(upper.preds.df,c(preds$upper[,1]))
    }

    #calculate accuracy
    #winkler: missing data error such as Turkey
    # w=winkler_cal(ts.entry$xx,preds$lower[,1],preds$upper[,1],0.2)
    # accuracy.df[(i-1),3]=mean(w)
  }
  point.preds.df=data.frame(point.preds.df)
  lower.preds.df=data.frame(lower.preds.df)
  upper.preds.df=data.frame(upper.preds.df)
  
  colnames(point.preds.df)=coln[2:length(coln)]
  colnames(lower.preds.df)=coln[2:length(coln)]
  colnames(upper.preds.df)=coln[2:length(coln)]
  do.call(grid.arrange,p)
  return(list(point.f=point.preds.df,
              lower.f=lower.preds.df,
              upper.f=upper.preds.df,
              ts.list=ts.list))
}


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#test
library(readxl)
library(forecast)
#data=read_excel('/Users/xixili/Downloads/TourismForecasting2023-main/TourismForecasting2023/Tourism\ forecasting\ competition\ II\ dataset.xlsx', sheet = "data for forecasting")
data=read_excel('Tourism\ forecasting\ competition\ II\ dataset.xlsx', sheet = "data for forecasting")


##################
train.end.date.post.covid='2023M07'
h=12
method.option='bagging'
averge.option=TRUE
res.post.covid=forecast.one.year.missing.proc(data,train.end.date.post.covid,h,method.option,averge.option)
#point forecasts of 20 countries or states
point.f.post.covid=res.post.covid$point.f
#loweer forecasts of 20 countries or states
lower.preds.post.covid=res.post.covid$lower.f
#upper forecasts of 20 countries or states
upper.preds.post.covid=res.post.covid$upper.f
#time series of 20 countries or states
ts.list=res.post.covid$ts.list
save(ts.list,file = 'ts.list.rda')
write.csv(data.frame(res.post.covid$point.f), file = "bagging_results.csv", row.names = TRUE)
write.csv(data.frame(lower.preds.post.covid), file = "lower_bound_xixi.csv", row.names = TRUE)
write.csv(data.frame(upper.preds.post.covid), file = "upper_bound_xixi.csv", row.names = TRUE)



