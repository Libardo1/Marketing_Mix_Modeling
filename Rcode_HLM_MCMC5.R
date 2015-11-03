rm(list=ls(all=TRUE))

library("data.table")
library("methods")
library("ggplot2")
library("MCMCpack")
library("tseries")
library("forecast")
library("lmtest")
library("glmnet")
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

# compute adj-R^2
f.adjR2 = function(actual, pred, Nobs, Nvar) {
    Rsquared = f.r2(actual, pred)
    adjR2 = 1 - (1-Rsquared)*(Nobs-1)/(Nobs-Nvar-1)
    return(adjR2)
}


###################################################################################
# change dir
setwd("MAINDIR")

## read data and remove na
data.havi = fread("MAINDIR/data/pmm_final_6.csv")
data.havi = data.frame(data.havi)
data.havi = na.omit(data.havi)

# x: define independent variables
regprice.column = c(35:49)
promoprice.column = c(50:64)
logRegularPrice.column = c(65:79)
logPromoprice.column = c(81:95)
logDiscount.column = c(96:109, 80)
HaviPriceRatio.column = c(111:125)
national.column = c(126:149)
tactic.column = c(150:189)
holiday.column = c(190:296)
weather.column = c(297:348)
dayweek.column = c(349:354)
weekyear.column = c(355:406)
biweek.column = c(407)
monthyear.column = c(408:418)
calYear.column = c(419:423)

# selected fix-effect variables -------------------------------------------------------------------
ix = c(logDiscount.column, logRegularPrice.column, national.column, tactic.column, holiday.column, weather.column, dayweek.column, weekyear.column, biweek.column, monthyear.column, calYear.column)
# selected random variables -----------------------------------------------------------------------
random.select = c(logDiscount.column, logRegularPrice.column)
#--------------------------------------------------------------------------------------------------

# dataset
logSales = log(data.havi[,c("XXX")])
dataset = cbind(storeID=data.havi$storeID, logSales=logSales, data.havi[,ix])
dataset = data.frame(dataset)


##################################################################################################
# RUN
store.list = unique(dataset$storeID)
NumStore = length(store.list)

# validation
set.seed(as.numeric(try(system("date +%H%M%S",intern = TRUE))))
r = vector()
repeats = 1
result = foreach(icount(repeats), .combine=rbind) %dopar% {

            #################################################################################
            # # Apply lasso to select the predictors
            x = model.matrix(logSales~., dataset[,-1])[,-1]
            y = dataset$logSales

            # Use cv.glmnet() to determine the best lambda for the lasso:
            mod.lasso = cv.glmnet(x=x, y=y, type.measure='mse', nfolds=10, alpha=1)
            (bestlam.lasso = mod.lasso$lambda.min)

            # grep the variables which have non-zero coefficients in lasso
            c = coef(mod.lasso, s=bestlam.lasso, exact=TRUE)
            inds = which(c!=0)
            variables = row.names(c)[inds]
            variables = variables[variables %ni% '(Intercept)']
            variables = unique(  c(names(data.havi[,c(random.select,
                                                      dayweek.column, 
						      weekyear.column, 
						      biweek.column, 
						      monthyear.column, 
						      calYear.column)]),
                                         variables ))

            fit.formula = as.formula(paste("logSales ~ ",paste(variables, collapse=" + ")))
            fit.formula
            mod = lm(fit.formula, data=dataset)
            summary(mod)

            # grep variables which have non-NA coefficients
            coef = data.frame(summary(mod)$coefficients)
            variables = rownames(coef)[-1]

            ##################################################################################################
            # formula of fixed-effect terms
            fix.variables = unique(c(  names(data.havi[,c(random.select)]), variables  ))
            fix.formula = as.formula(paste("logSales ~ 1 + ",paste(fix.variables, collapse=" + ")))
            #fix.formula
            #length(fix.variables)

            # formula of random-effect terms
            random.variables = c(names(data.havi[,c(random.select)]))
            random.formula = as.formula(paste("~ 1 + ",paste(random.variables, collapse=" + ")))
            #random.formula

            rm(x, y, c, inds, coef, variables)


            #################################################################################
            # # MCMC/HLM simulation
            seed = as.numeric(try(system("date +%H%M%S",intern = TRUE)))
            nvar = length(random.variables) + 1
            nvardiag = nvar*diag(nvar)

            model = MCMChregress(fixed = fix.formula,
                                 random = random.formula,
                                 group = "storeID",
                                 data = dataset,
                                 #### parameters
                                 burnin = 1000,   # burnin iterations
                                 mcmc = 10000,    # Gibbs iterations
                                 thin = 5,        # thinning interval
                                 r = nvar,        # shape parameter for the Inverse-Wishart
                                 R = nvardiag,    # scale matrix for the Inverse-Wishart
                                 verbose = 1, seed = seed, beta.start = 0, sigma2.start = 1, mubeta = 0, Vbeta = 1e+06,
                                 nu = 0.001, delta = 0.001)

            # overall R^2
            R2 = f.r2(dataset$logSales, model$Y.pred)
            adjR2 = f.adjR2(dataset$logSales, model$Y.pred, Nobs=nrow(dataset), Nvar=length(fix.variables)*NumStore)

            # RMSE
            RMSE = f.rmse(dataset$logSales, model$Y.pred)

            print(paste0("RMSE = ", RMSE))
            print(paste0("R-square = ", R2))
            print(paste0("adjusted R-square = ", adjR2))

            # store-level performance
            output.y = cbind(storeID=dataset$storeID,
                             actual=dataset$logSales,
                             pred=model$Y.pred,
                             residuals=dataset$logSales-model$Y.pred)
            output.y = data.frame(output.y)
            write.csv(output.y, "WORKDIR/XXX.y.csv")

            result.store = matrix(NA, NumStore, 4)
            for (store in 1:NumStore) {
                temp = subset(output.y, storeID==store.list[store])
                store.r2 = f.r2(temp$actual, temp$pred)
                store.adjr2 = f.adjR2(temp$actual, temp$pred, Nobs=nrow(temp), Nvar=length(fix.variables))
                box.test.p = Box.test(temp$residuals, lag=nrow(temp), fitdf=length(fix.variables), type="Ljung")$p.value

                result.store[store, 1] = store.list[store]
                result.store[store, 2] = store.r2
                result.store[store, 3] = store.adjr2
                result.store[store, 4] = box.test.p
            }

            result.store = data.frame(result.store)
            colnames(result.store) = c("storeID", "R2", "adj-R2", "Box p-value")
            result.store
            write.csv(result.store, "WORKDIR/XXX.store.performance.csv", row.names=FALSE)

            # pull coefficients --------------------------------------------------------------------------------------------
            print("Pull coefficients......")
            mcmc.statistics = data.frame(summary(model$mcmc)$statistics)

            # coefficients of the fixed-effect terms
            fix.coef = mcmc.statistics[grepl("beta.", rownames(mcmc.statistics), fixed=TRUE),][,1]

            # coefficients of the random-effect terms for each store
            random.variables2 = c("(Intercept)", random.variables)
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
            write.csv(coef.store,"WORKDIR/XXX.mcmc.coef.csv")

            return(c(R2=R2,
                     adjR2=adjR2,
                     RMSE=RMSE))

         }

result = data.frame(result)
write.csv(result,"WORKDIR/XXX.result.csv", row.names=FALSE)

stopCluster(cl)
