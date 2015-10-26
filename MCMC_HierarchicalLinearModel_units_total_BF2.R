rm(list=ls(all=TRUE))

library("data.table")
library("methods")
library("ggplot2")
library("MCMCpack")
library("nlme")
library("coda")
library("doParallel")
# Find out how many cores are available (if you don't already know)
NumNode = detectCores(); NumNode
# Create cluster with desired number of cores
cl = makeCluster(1)
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

# compute RMSE
f.rmse = function(actual, pred) {
    rmse = sqrt(mean((actual-pred)^2))
    return(rmse)
}

# compute R2
f.r2 = function(actual, pred) {
    r2 = 1 - (sum((actual-pred)^2)/sum((actual-mean(actual))^2))
    return(r2)
}


###################################################################################
# change dir
setwd("/home/linpyl/HAVI")

## read data and remove na
data.havi = fread("/home/linpyl/HAVI/data/pmm_final_3.csv")
data.havi = data.frame(data.havi)
data.havi = na.omit(data.havi)
#dim(data.havi)
# remove the observations of storeID=96, 97, 98
#data.havi = data.havi[data.havi$storeID!=96 & data.havi$storeID!=97 & data.havi$storeID!=98,]
#dim(data.havi)

SearchIndex = fread("/home/linpyl/HAVI/data/SearchIndex.csv")

# define variable types
ix = c(1,35:317)
#data.havi[ix] = lapply(data.havi[ix], as.factor)

# y: name of product
product.index = SearchIndex[product_name %in% "units_total_BF2"]$index 

# x: selected independent variables
dayweek.column = c(36:41)
holiday.column = c(42:44, 46:51, 53:58, 60:86, 88:93, 95:100, 102:111, 113:116, 118:128, 130:134, 136:141, 143:148, 150:155, 157:158, 160:162)
weather.column = c(163:214)
national.column = c(219:242)
tactic.column = c(244:249, 251:254, 256:259, 261:266, 268:269, 272:273, 276:278, 283:284, 288:289, 298:299, 304, 306:309, 314, 317)
price.column = c(318:332)
promoprice.column = c(333,335,337,339,341,343,345,347,349,351,353,355,357,359,361)
regprice.column = c(334,336,338,340,342,344,346,348,350,352,354,356,358,360,362)
logDiscount.column = c(378:392)
logRegularPrice.column = c(363:377)
weekyear.column = c(394:445)

ix = c(dayweek.column, holiday.column, weather.column, national.column, tactic.column, logDiscount.column, logRegularPrice.column, weekyear.column)

# selected random variables -----------------------------------------------------------------------
random.select = c(logDiscount.column, logRegularPrice.column)
#--------------------------------------------------------------------------------------------------

# dataset
logSales = log(data.havi[,c(product.index)])
dataset = cbind(storeID=data.havi$storeID, logSales=logSales, data.havi[,ix])
dataset = data.frame(dataset)


##################################################################################################
# formula of fixed-effect terms
fix.variables = names(dataset[,c(-1,-2)])
fix.formula = as.formula(paste("logSales ~ 1 + ",paste(fix.variables, collapse=" + ")))
print("formula of fixed-effect terms: ")
fix.formula

# formula of random-effect terms
random.variables = names(data.havi[,random.select])
random.formula = as.formula(paste("~ 1 + ",paste(random.variables, collapse=" + ")))
print("formula of random-effect terms: ")
random.formula

# RUN
nvar = length(random.variables) + 1
nvardiag = nvar*diag(nvar)
# validation
set.seed(as.numeric(try(system("date +%H%M%S",intern = TRUE))))
r = vector()
repeats = 1
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
            #data.train = data.train[,-1]   # remove the column of storeID
            #data.test = data.test[,-1]   # remove the column of storeID
            rm(temp, temp.train, temp.test, data.storeID, store)

            seed = as.numeric(try(system("date +%H%M%S",intern = TRUE)))
            model = MCMChregress(fixed = fix.formula,
                     		 random = random.formula,
                     		 group = "storeID",
                     		 data = data.train,
                     		 #### parameters
                     		 burnin = 1000,   # burnin iterations
                     		 mcmc = 10000,    # Gibbs iterations
                     		 thin = 5,        # thinning interval
                     		 r = nvar,        # shape parameter for the Inverse-Wishart
                     		 R = nvardiag,    # scale matrix for the Inverse-Wishart
                     		 verbose = 1, seed = seed, beta.start = 0, sigma2.start = 1, mubeta = 0, Vbeta = 1e+06,
                     		 nu = 0.001, delta = 0.001)

            # R^2
	        train.R2 = f.r2(data.train$logSales, model$Y.pred)
	        print(paste0("in-sample R-square = ", train.R2))

            # pull coefficients --------------------------------------------------------------------------------------------
            print("Pull coefficients......")
            mcmc.statistics = data.frame(summary(model$mcmc)$statistics)

            # coefficients of the fixed-effect terms
	        fix.coef = mcmc.statistics[grepl("beta.", rownames(mcmc.statistics), fixed=TRUE),][,1]

            # coefficients of the random-effect terms for each store
	        random.variables2 = c("(Intercept)", random.variables)
	        store.list = unique(dataset$storeID)
	        random.coef = matrix(NA, nrow=length(random.variables2), ncol=length(store.list))
	        for (i in 1:NumStore) {
    		    for (j in 1:length(random.variables2)) {
        	        random.coef[j,i] = mcmc.statistics[c(paste0("b.",random.variables2[j],".",store.list[i])),][,1]
    		    }
	        }

	        # Array storing all coefficients for each store
	        coef.storeID = matrix(NA, nrow=length(fix.variables)+length(random.variables)+2, ncol=length(store.list))
	        random.start = length(fix.coef)+1
	        random.end = length(fix.variables)+length(random.variables)+2
	        temp = length(fix.coef)+1
	        for (i in 1:length(store.list)) {
    		    coef.storeID[1:length(fix.coef),i] = fix.coef
    		    coef.storeID[random.start:random.end,i] = random.coef[,i]
	        }
	        rownames(coef.storeID) = c("beta.(Intercept)",paste0("beta.",fix.variables),"b.(Intercept)",paste0("b.",random.variables))
	        colnames(coef.storeID) = c(paste0(store.list))
	        print(coef.storeID)
             
            # save coefficients
            coef.name = c("(Intercept)",names(dataset[,c(-1,-2)]))
            coef.store = matrix(NA, nrow=length(coef.name), ncol=length(store.list))
            coef.store = data.frame(coef.store)
            rownames(coef.store) = coef.name
            colnames(coef.store) = c(store.list)

            for (i in 1:length(store.list)) {
                for (j in 1:length(coef.name)) {
                    temp.coef.name = coef.name[j]
                    coef.store[j,i] = sum(coef.storeID[which(rownames(coef.storeID)==paste0("beta.",temp.coef.name)
                                                       | rownames(coef.storeID)==paste0("b.",temp.coef.name)),i])
                }
            }

            #print(coef.store)
            write.csv(coef.store,"/home/linpyl/HAVI/MMM_f5_logDiscount_logRegPrice/units_total_BF2.csv")

            # --------------------------------------------------------------------------------------------------------------
            # my train.R2
            data.storeID = split(data.train, data.train$storeID)
            table.my.r2 = data.frame()

            for (store in 1:NumStore) {
                temp = data.frame(data.storeID[[store]])
                temp2 = cbind(I.fix=rep(1,nrow(temp)), temp[,c(-1,-2)])
                pred = as.matrix(temp2) %*% coef.store[,store]
                table.my.r2 = rbind(table.my.r2, cbind(temp[,2],pred))
            }

            table.my.r2 = data.frame(table.my.r2)
            my.R2 = f.r2(table.my.r2[,1], table.my.r2[,2])
            print(paste0("my in-sample R-square = ", my.R2))
            print(paste0("in-sample R-square = ", train.R2))

            rm(data.storeID)

            # --------------------------------------------------------------------------------------------------------------
            # compute test-RMSE
            print("Compute RMSE for test set......")
            data.storeID = split(data.test, data.test$storeID)
            table.rmse = vector()
            table.holdout.r2 = data.frame()

            for (store in 1:NumStore) {
                temp = data.frame(data.storeID[[store]])
                temp2 = cbind(I.fix=rep(1,nrow(temp)), temp[,c(-1,-2)])
                pred = as.matrix(temp2) %*% coef.store[,store]
                table.holdout.r2 = rbind(table.holdout.r2, cbind(temp[,2],pred))
                err = temp[,2] - pred
                table.rmse = c(table.rmse, err)
            }

            RMSE = sqrt(mean(table.rmse^2))
            print(paste0("test-RMSE = ", RMSE))

            # holdout R^2
            table.holdout.r2 = data.frame(table.holdout.r2)
            test.R2 = f.r2(table.holdout.r2[,1], table.holdout.r2[,2])
            print(paste0("holdout R-square = ", test.R2))


            return(c(train.R2=train.R2,
                     test.R2=test.R2,
                     RMSE=RMSE))
         }

result = data.frame(result)
write.csv(result,"/home/linpyl/HAVI/MMM_f5_logDiscount_logRegPrice/units_total_BF2.result.csv", row.names=FALSE)

stopCluster(cl)
