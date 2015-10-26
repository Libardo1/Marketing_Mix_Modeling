rm(list=ls(all=TRUE))

library("data.table")
library("methods") 
library("foreach")
library("doParallel")
# Find out how many cores are available (if you don't already know)
NumNode = detectCores(); NumNode
# Create cluster with desired number of cores
cl = makeCluster(3)
# Register cluster
registerDoParallel(core=cl)
# Find out how many cores are being used
getDoParWorkers()


##############################################################################
# FUNCTIONS
#
# function to handle the situation where function returning more than one value
':=' = function(lhs, rhs) {
    frame = parent.frame()
    lhs = as.list(substitute(lhs))
    if (length(lhs) > 1)
		lhs = lhs[-1]
	if (length(lhs) == 1) {
		do.call(`=`, list(lhs[[1]], rhs), envir=frame)
		return(invisible(NULL)) }
	if (is.function(rhs) || is(rhs, 'formula'))
		rhs = list(rhs)
	if (length(lhs) > length(rhs))
		rhs = c(rhs, rep(list(NULL), length(lhs) - length(rhs)))
	for (i in 1:length(lhs))
		do.call(`=`, list(lhs[[i]], rhs[[i]]), envir=frame)
	return(invisible(NULL)) 
}

# Split function to split data set into training and test sub sets:
f.split = function(data, probtrain, probtest) {
    # label training and test data sets
    index = sample(x=c(0,1), size=nrow(data), prob=c(probtrain,probtest), replace=TRUE)
    # split data set into training and test
    train = data[index==0,]
    test = data[index==1,]  
    return(list(train, test))
}

#
`%ni%` = Negate(`%in%`)


###################################################################################
# change dir
setwd("/home/linpyl/HAVI")

## read data and remove na
data.havi = fread("/home/linpyl/HAVI/data/pmm_final_3.csv")
print("read data set already")
data.havi = data.frame(data.havi)
data.havi = na.omit(data.havi)
# remove the observations of storeID=96, 97, 98
data.havi = data.havi[data.havi$storeID!=96 & data.havi$storeID!=97 & data.havi$storeID!=98,]

SearchIndex = fread("/home/linpyl/HAVI/data/SearchIndex.csv")

# define variable types
ix = c(1,35:317) 
#data.havi[ix] = lapply(data.havi[ix], as.factor) 

# y: name of product
product.index = SearchIndex[product_name %in% "units_total_BF2"]$index

# x: selected predictors
dayweek.column = c(36:41)
holiday.column = c(42:44, 46:51, 53:58, 60:86, 88:93, 95:100, 102:111, 113:116, 118:128, 130:134, 136:141, 143:148, 150:155, 157:158, 160:162)
weather.column = c(163:214)
national.column = c(219:242)
tactic.column = c(244:249, 251:254, 256:259, 261:266, 268:269, 272:273, 276:278, 283:284, 288:289, 298:299, 304, 306:309, 314, 317)
Haviprice.column = c(318:332)
promoprice.column = c(333,335,337,339,341,343,345,347,349,351,353,355,357,359,361)
regprice.column = c(334,336,338,340,342,344,346,348,350,352,354,356,358,360,362)
logDiscount.column = c(378:392)
logRegularPrice.column = c(363:377)
weekyear.column = c(394:445)
        
ix = c(dayweek.column, holiday.column, weather.column, national.column, tactic.column, logDiscount.column, logRegularPrice.column, weekyear.column)
        
logSales = log(data.havi[,c(product.index)])
dataset = cbind(storeID=data.havi$storeID, logSales=logSales, data.havi[,ix])
dataset = data.frame(dataset)


############################################################################################################
# run
mod = glm(logSales ~., data=dataset[,-1])
summary(mod)

# validation
set.seed(1234)
r = vector()
repeats = 100
result = foreach(icount(repeats), .combine=rbind) %dopar% {
            # split data set based on storeID and then merge back
            data.storeID = split(dataset, dataset$storeID)
            NumStore = length(names(data.storeID))
            data.train <- data.test <- vector()
            for (store in 1:NumStore) {
                temp = data.frame(data.storeID[[store]])
            	c(temp.train, temp.test) := f.split(temp, probtrain=0.7, probtest=0.3)
            	data.train = rbind(data.train, temp.train)
            	data.test = rbind(data.test, temp.test)
        	}
            data.train = data.frame(data.train)
            data.test = data.frame(data.test)
            data.train = data.train[,-1]   # remove the column of storeID
            data.test = data.test[,-1]   # remove the column of storeID
            rm(temp, temp.train, temp.test, data.storeID, store, NumStore)

            mod = glm(logSales ~., data=data.train)

            train.R2 = 1 - (sum((data.train$logSales-predict(mod))^2)/sum((data.train$logSales-mean(data.train$logSales))^2))
            test.R2 = 1 - (sum((data.test$logSales-predict(mod, newdata=data.test))^2)/sum((data.test$logSales-mean(data.test$logSales))^2))
 
            return(c(mod$coefficients,
                     train.R2=train.R2,
                     test.R2=test.R2,
                     AIC=AIC(mod),
                     BIC=BIC(mod),
                     RMSE=sqrt(mean((data.test$logSales-predict(mod, newdata=data.test))^2))))
         } 

result = data.frame(result)
write.csv(result,"/home/linpyl/HAVI/GLM_fullModel_validation_f5/units_total_BF2.result1.csv", row.names=FALSE)

# compute mean, sd 
result.mean = apply(result, 2, mean)
result.sd = apply(result, 2, sd)
result2 = rbind(names(result), mean=result.mean, sd=result.sd)
result2 = data.frame(result2)
write.csv(result2,"/home/linpyl/HAVI/GLM_fullModel_validation_f5/units_total_BF2.result2.csv", row.names=FALSE)

# print outputs
# in-sample R^2
print(paste0("mean in-sample R-squared = ", round(mean(result$train.R2),4)))
print(paste0("sd in-sample R-squared = ", round(sd(result$train.R2),4)))

# holdout R^2
print(paste0("mean holdout R-squared = ", round(mean(result$test.R2),4)))
print(paste0("sd holdout R-squared = ", round(sd(result$test.R2),4)))

# AIC and BIC
print(paste0("mean AIC = ", round(mean(result$AIC),3)))
print(paste0("sd AIC = ", round(sd(result$AIC),3)))
print(paste0("mean BIC = ", round(mean(result$BIC),3)))
print(paste0("sd BIC = ", round(sd(result$BIC),3)))

# rmse
print(paste0("mean RMSE = ", round(mean(result$RMSE),4)))
print(paste0("sd RMSE = ", round(sd(result$RMSE),4)))

stopCluster(cl)
