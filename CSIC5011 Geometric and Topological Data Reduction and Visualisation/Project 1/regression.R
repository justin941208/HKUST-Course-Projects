#####regression
library(fUnitRoots)
library("forecast")
library(DAAG) 
library(forecast)
library(stats)
library(lmtest)
a<-read.table("D:/evdata/mini_project/topic_final",sep=" ")
cor(ts(a))
x1 <- a[,1]
x2 <- a[,2]
x3 <- a[,3]
x4 <- a[,4]
x5 <- a[,5]
mydata <-data.frame(x1,x2,x3,x4,x5)
#linear regression
lm.sol<-lm(x1~x5+x2+x4+x3,data=mydata)
summary(lm.sol)
confint(lm.sol)
y.res<-residuals(lm.sol)

#boxcox
##boxcox(y~x1+x2+x3+x4,data=mydata)
BoxCox.lambda(x1, method = "loglik") 
x5 <- BoxCox(x1, lambda = 0.05)

##stepwise regression
lm.sol= step(lm.sol)
#
par(mfrow=c(2,2))  
plot(lm.sol) 

#student residue
y.stu<-rstudent(lm.sol)  
y.fit<-predict(lm.sol)  
par(mfrow=c(2,1))  
plot(y.stu~y.fit)  
hist(y.stu,freq=FALSE)  
lines(density(y.stu)) 

#test multi collinearity
kappa(cor(ts(a)))  
vif(lm.sol)   
###test heteroscedasticity and remove ot
bptest(lm.sol)
gqtest(lm.sol)
lm.test2<-lm(log(resid(lm.sol)^2)~x1+x2+x3+x4,data=mydata)
lm.test3<-lm(y~x1+x2+x3+x4,weights=1/exp(fitted(lm.test2)),data=mydata)
summary(lm.test3)
bptest(lm.test3)
lm.sol = lm.test3


