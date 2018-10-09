########forecast
library(fUnitRoots)
library("forecast")
a1<-read.table("D:/evdata/mini_project/topic_final",sep=" ")
worker<-a1[,1]
worker<-ts(worker)
worker<-a1[,2]
worker<-a1[,3]
worker<-a1[,4]
worker<-a1[,5]
##seqquence diagram
par(mfrow=c(3,2))
worker<-ts(worker)
plot(worker,main='topic5 sequence diagram',ylab='probability')

###difference1 #diff(1:10, lag=3)
worker<-a1[,1]
worker<-ts(worker)
worker.dif<-diff(worker,differences=3) #three level   ## good
adfTest(worker.dif,lag=1,type="ct") #adf test

###define parameter of arima model
par(mfrow=c(2,1))
acf(worker.dif,lag.max=50)
pacf(worker.dif,lag.max=30)

###acf trailing£¬pacf truncation select model AR+c(na,0,0,na)

###simulate model1
worker.fit<-arima(x=worker,order=c(12,3,0),seasonal = list(order = c(1,1,0),
                 period=1),transform.pars=F)
worker.fore<-forecast(worker.fit,h=3)
plot(worker.fore)

##
Box.test(worker.fore$residuals, lag=5, type="Ljung-Box")#test residue is white noise
plot.ts(worker.fore$residuals)





