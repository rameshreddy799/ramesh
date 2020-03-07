setwd("C:/Users/Satish/Desktop/MyFiles/Data_Analytics/data")

df = read.csv("Churn_MV.csv")

head(df)

df1 <- df[is.na(df$Churn)==FALSE,]

df<-df1

### Check Class imbalance 

table( df$Churn) ## freq table 

prop.table(table( df$Churn) )

#### Data transformation

str(df)

df$Churn <- as.factor(df$Churn)
df$Intl.Plan <- as.factor(df$Intl.Plan)
df$VMail.Plan <- as.factor(df$VMail.Plan)


df$State <- as.factor(df$State)
df$Area.Code  <- as.factor(df$Area.Code )
df$Phone <- as.factor(df$Phone)

### Data seperate

df_num <- df[,-c(8,9,10,20:22)]

df_cat <- df[,c(8,9,10,20:22)]

colnames(df_num)

#### Null Imputation

summary(df_num)

mean(df_num$Daily.Charges.MV,na.rm = T)
median(df_num$Daily.Charges.MV,na.rm = T)


df_num$Daily.Charges.MV_mean <- ifelse(is.na(df_num$Daily.Charges.MV)==T,
                                       mean(df_num$Daily.Charges.MV,na.rm = T),df_num$Daily.Charges.MV)

df_num$Daily.Charges.MV_median <- ifelse(is.na(df_num$Daily.Charges.MV)==T,
                                         median(df_num$Daily.Charges.MV,na.rm = T),df_num$Daily.Charges.MV)



### rmse


# Function that returns Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Function that returns Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}


#### error calculating

error <- df_num$Day.Charge-df_num$Daily.Charges.MV_mean 

rmse(error) ####1.50565

error <- df_num$Day.Charge-df_num$Daily.Charges.MV_median

rmse(error)  #### 1.506297

### outlier

boxplot(df_num$Day.Mins)


x <- df_num$Day.Mins
qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
caps <- quantile(x, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(x, na.rm = T)
x[x < (qnt[1] - H)] <- caps[1]
x[x > (qnt[2] + H)] <- caps[2]

df_num$Day.Mins <- ifelse(df_num$Day.Mins <  (qnt[1] - H),caps[1],df_num$Day.Mins)
df_num$Day.Mins <- ifelse(df_num$Day.Mins > (qnt[2] + H),caps[2],df_num$Day.Mins)

boxplot(df_num$Eve.Calls)

x <- df_num$Eve.Calls
qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
caps <- quantile(x, probs=c(.05, .95), na.rm = T)
H <- 1.5 * IQR(x, na.rm = T)
x[x < (qnt[1] - H)] <- caps[1]
x[x > (qnt[2] + H)] <- caps[2]

df_num$Eve.Calls <- ifelse(df_num$Eve.Calls <  (qnt[1] - H),caps[1],df_num$Eve.Calls)
df_num$Eve.Calls <- ifelse(df_num$Eve.Calls> (qnt[2] + H),caps[2],df_num$Eve.Calls)

### scaling

df_num_zscore <- as.data.frame(scale(df_num))

summary(df_num_zscore)

fun_min_max <-   function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}

colnames(df_num)

df_num_min_max <- as.data.frame(lapply(df_num, fun_min_max))

str(df_num_min_max)

#### correlation
install.packages("corrplot")
library(corrplot)

cor(df_num)

corrplot(cor(df_num), method = "shade")

#### cat - cat
str(df)

chisq.test(df$Intl.Plan, df$VMail.Plan, correct=FALSE)


### cat- Num 

test_cat_num <- aov(df$VMail.Messages~ df$Intl.Plan)


### outpu calling of anova

summary(test_cat_num)
# 
# sample :
#   Df Sum Sq Mean Sq F value  Pr(>F)   
# fac          3   1636   545.5   5.406 0.00688 **
#   Residuals   20   2018   100.9                   
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 


df_final <- cbind(df_cat,df_num)

df_final_1 <- cbind(df_cat,df_num_min_max)

df_final_2 <- cbind(df_cat,df_num_zscore)





#### EDA to select the most important variables 

df$Churn = as.factor( df$Churn)

#### relationship b/w Churn and Day.charges 
library(ggplot2)
ggplot( df, aes( Churn, Day.Charge)) + geom_boxplot()

ggplot( df, aes( Churn, Eve.Charge)) + geom_boxplot()

ggplot( df, aes( Churn, Night.Charge)) + geom_boxplot()

### cor of day charges with night and evening 

cor(df$Day.Charge, df$Day.Mins)
cor( df$Eve.Charge, df$Night.Charge)

### explore with factor variables 

ggplot( df, aes(VMail.Plan, fill = Churn )) + geom_bar(position = "fill")

ggplot( df, aes( Intl.Plan, fill = Churn)) + geom_bar()

### convert Vmailplan and Intl.plan to factor variables 

df$VMail.Plan = as.factor(df$VMail.Plan)
df$Intl.Plan = as.factor(df$Intl.Plan)

table(df$VMail.Plan)

## divide the dataset into training and test set 

set.seed( 1234)

ids = sample( nrow(df_num), nrow(df)*0.8)

train = df[ ids,]
test =  df[-ids,]


##### linear regression 

linearMod <- lm(CustServ.Calls  ~ .-Daily.Charges.MV_mean-Daily.Charges.MV_median-Daily.Charges.MV, data=train) 


summary(linearMod)
AIC(linearMod) 

Pred <- predict(linearMod, test) 

actuals_preds <- data.frame(cbind(actuals=test$CustServ.Calls, predicteds=Pred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  
head(actuals_preds)

mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals) 

install.packages('DMwR')

library(DMwR)

regr.eval(actuals_preds$actuals, actuals_preds$predicteds)

### startified sample

df_lin <- cbind (df_num,df$VMail.Plan)
library(caTools)
train_rows = sample.split(df$VMail.Plan, SplitRatio=0.9)
train = data[ train_rows,]
test  = data[!train_rows,]


##### linear regression 

linearMod <- lm(CustServ.Calls  ~ .-Daily.Charges.MV_mean-Daily.Charges.MV_median-Daily.Charges.MV, data=train) 

summary(linearMod)

AIC(linearMod)


Pred <- predict(linearMod, test) 

actuals_preds <- data.frame(cbind(actuals=test$CustServ.Calls, predicteds=Pred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  
head(actuals_preds)

mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals) 

install.packages('DMwR')

library(DMwR)

regr.eval(actuals_preds$actuals, actuals_preds$predicteds)



******* logstic

### startified sample
library(caTools)
train_rows = sample.split(df$Churn, SplitRatio=0.7)
train = data[ train_rows,]
test  = data[!train_rows,]


churn_model = glm( Churn ~ . -State-Area.Code-Phone-Daily.Charges.MV, 
                   data = train, 
                   family="binomial")


summary(churn_model)


### Predict the test observations 

test$pred = predict( churn_model , newdata = test, type="response")


test$pred_class = ifelse( test$pred >= 0.5, 1 , 0)

### confusion matrix 
table(test$pred_class, test$Churn )
## ROC graphs 

library(ROCR)

### add the ROC graph of credit_model1 on the same plot 
pred = prediction(test$pred , test$Churn)
perf= performance(pred, "tpr","fpr")
plot(perf)
### AUC for churn model 

AUC_1 = performance(pred, measure = 'auc')@y.values[[1]]
AUC_1


### precison vs Recall curve 

library("DMwR")

PRcurve(test$pred, test$Churn)




### change the cutoff 

test$pred_class2 = ifelse( test$pred >=0.3, 1, 0)

table( test$pred_class2, test$Churn)

**********************************************************
#####Decision Tree 
library(rpart)

### startified sample
library(caTools)
train_rows = sample.split(df$Churn, SplitRatio=0.7)
train = data[ train_rows,]
test  = data[!train_rows,]

fitTree<-rpart(Churn ~.-State-Area.Code-Phone-Daily.Charges.MV, 
                   data = train)


test$pred =   predict(fitTree, test,type="class")
				   
### confusion matrix 
table(test$pred, test$Churn )				   

#### randome forest

library(party)
library(randomForest)

# Create the forest.
output_forest <- randomForest(Churn ~.-State-Area.Code-Phone-Daily.Charges.MV, 
                   data = train)

				   

test$pred =   predict(output_forest, test)
				   
### confusion matrix 
table(test$pred, test$Churn )

### clustering


df_Clust <- df[,c(20:22)]

freq_table <- as.data.frame(table(df_Clust$State,df_Clust$Area.Code))

freq_table <- as.matrix(freq_table)
clust<- akmeans(freq_table) 

clust$cluster

clust$size

df_Clust_1<- cbind(df_Clust,clust$cluster)


df_after_Clust <- merge(df,df_Clust_1,by.x=c("State","Area.Code"),by.y=c("df_Clust$State","df_Clust$Area.Code"))




### alternative approach


wssplot <- function(freq_table, nc=15, seed=1234){
               wss <- (nrow(data)-1)*sum(apply(freq_table,2,var))
               for (i in 2:nc){
                    set.seed(seed)
                    wss[i] <- sum(kmeans(freq_table, centers=i)$withinss)}
                plot(1:nc, wss, type="b", xlab="Number of Clusters",
                     ylab="Within groups sum of squares")}
					 
clust2 <- kmeans(freq_table, 15, nstart=25)  

clust2$size

clust2$cluster



df_Clust_1<- cbind(df_Clust,clust2$cluster)


df_after_Clust <- merge(df,df_Clust_1,by.x=c("State","Area.Code"),by.y=c("df_Clust$State","df_Clust$Area.Code"))




#### randome forest

library(caTools)
train_rows = sample.split(df_after_Clust$Churn, SplitRatio=0.7)
train = df_after_Clust[ train_rows,]
test  = df_after_Clust[!train_rows,]

library(party)
library(randomForest)

# Create the forest.
output_forest <- randomForest(Churn ~.-State-Area.Code-Phone-Daily.Charges.MV, 
                   data = train)

				   

test$pred =   predict(output_forest, test)
				   
### confusion matrix 
table(test$pred, test$Churn )


  
#### build random forest +Logistic
  
  install.packages("randomForest")
library(randomForest)

churn_model_rf = randomForest(( Churn ~ ., 
                                data = df)
                              
                              importance(churn_model_rf)        
                              Top <- varImpPlot(churn_model_rf)   
                              
                              ### Top 15 consider variables to build the model
                              
                              Churn_model_rf1 <- churn_model_rf[,c(Top)]
                              
                              
                              
                              ### startified sample
                              library(caTools)
                              train_rows = sample.split(Churn_model_rf1$Churn, SplitRatio=0.7)
                              train = data[ train_rows,]
                              test  = data[!train_rows,]
                              
                              
                              #### build logistic regression model with randomforest
                              
                              model_15 = glm( Churn ~ ., 
                                              data = train, 
                                              family="binomial")
                              
                              
                              
                              
                              ### Predict the test observations 
                              
                              test$pred = predict( model_15 , newdata = test, type="response")
                              
                              
                              test$pred_class = ifelse( test$pred >= 0.5, 1 , 0)
                              
                              ### confusion matrix 
                              table(test$pred_class, test$Churn )
                              
                              
                              ###### 
							  
#### build random forest +Logistic +clustering


churn_model_rf = randomForest(( Churn ~ ., 
                                data = df_after_Clust)
                              
                              importance(churn_model_rf)        
                              Top <- varImpPlot(churn_model_rf)   
                              
                              ### Top 15 consider variables to build the model
                              
                              Churn_model_rf1 <- churn_model_rf[,c(Top)]
                              
                              
                              
                              ### startified sample
                              library(caTools)
                              train_rows = sample.split(df_after_Clust$Churn, SplitRatio=0.7)
                              train = df_after_Clust[ train_rows,]
                              test  = df_after_Clust[!train_rows,]
                              
                              
                              #### build logistic regression model with randomforest
                              
                              model_15 = glm( Churn ~ ., 
                                              data = train, 
                                              family="binomial")
                              
                              
                              
                              
                              ### Predict the test observations 
                              
                              test$pred = predict( model_15 , newdata = test, type="response")
                              
                              
                              test$pred_class = ifelse( test$pred >= 0.5, 1 , 0)
                              
                              ### confusion matrix 
                              table(test$pred_class, test$Churn )
                              
## PCA 

PCA1 <- prcomp(df_num_min_max, scale. = T)

std_1 <- PCA1$sdev

var_1 <- std_1^2

plot_cum <- var_1/sum(var_1)

plot(cumsum(plot_cum), xlab = "Principal Component",
              ylab = "Cumulative Proportion of Variance Explained",
              type = "b")

			  pca_2 <- s.data.frame(PCA1$x)
			  
			  pca_3 <- pca_2[,c(1:5)]
			  
			  
			  
pca_df <- cbind(df_cat,pca_3)



******* logstic

### startified sample
library(caTools)
train_rows = sample.split(pca_df$Churn, SplitRatio=0.7)
train = pca_df[ train_rows,]
test  = pca_df[!train_rows,]


churn_model_pca = glm( Churn ~ . -State-Area.Code-Phone, 
                   data = train, 
                   family="binomial")


summary(churn_model)


### Predict the test observations 

test$pred = predict( churn_model_pca , newdata = test, type="response")


test$pred_class = ifelse( test$pred >= 0.5, 1 , 0)

### confusion matrix 
table(test$pred_class, test$Churn )
## ROC graphs 

library(ROCR)

### add the ROC graph of credit_model1 on the same plot 
pred = prediction(test$pred , test$Churn)
perf= performance(pred, "tpr","fpr")
plot(perf)
### AUC for churn model 

AUC_1 = performance(pred, measure = 'auc')@y.values[[1]]
AUC_1


### precison vs Recall curve 

library("DMwR")

PRcurve(test$pred, test$Churn)




### change the cutoff 

test$pred_class2 = ifelse( test$pred >=0.8, 1, 0)

table( test$pred_class2, test$Churn)


## PCA + RF+Logistic


churn_model_rf = randomForest(( Churn ~ ., 
                                data = pca_df)
                              
                              importance(churn_model_rf)        
                              Top <- varImpPlot(churn_model_rf)   
                              
                              ### Top 15 consider variables to build the model
                              
                              Churn_model_rf1 <- churn_model_rf[,c(Top)]
                              
                              
                              
                              ### startified sample
                              library(caTools)
                              train_rows = sample.split(pca_df$Churn, SplitRatio=0.7)
                              train = pca_df[ train_rows,]
                              test  = pca_df[!train_rows,]
                              
                              
                              #### build logistic regression model with randomforest
                              
                              model_15 = glm( Churn ~ ., 
                                              data = train, 
                                              family="binomial")
                              
                              
                              
                              
                              ### Predict the test observations 
                              
                              test$pred = predict( model_15 , newdata = test, type="response")
                              
                              
                              test$pred_class = ifelse( test$pred >= 0.5, 1 , 0)
                              
                              ### confusion matrix 
                              table(test$pred_class, test$Churn )
							  
							  
##PCA+Clust+RF+logistic
df_after_Clust_pca <- merge(pca_df,df_Clust_1,by.x=c("State","Area.Code"),by.y=c("df_Clust$State","df_Clust$Area.Code"))



churn_model_rf = randomForest(( Churn ~ ., 
                                data = df_after_Clust_pca)
                              
                              importance(churn_model_rf)        
                              Top <- varImpPlot(churn_model_rf)   
                              
                              ### Top 15 consider variables to build the model
                              
                              Churn_model_rf1 <- churn_model_rf[,c(Top)]
                              
                              
                              
                              ### startified sample
                              library(caTools)
                              train_rows = sample.split(df_after_Clust_pca$Churn, SplitRatio=0.7)
                              train = df_after_Clust_pca[ train_rows,]
                              test  = df_after_Clust_pca[!train_rows,]
                              
                              
                              #### build logistic regression model with randomforest
                              
                              model_15 = glm( Churn ~ ., 
                                              data = train, 
                                              family="binomial")
                              
                              
                              
                              
                              ### Predict the test observations 
                              
                              test$pred = predict( model_15 , newdata = test, type="response")
                              
                              
                              test$pred_class = ifelse( test$pred >= 0.5, 1 , 0)
                              
                              ### confusion matrix 
                              table(test$pred_class, test$Churn )

## REplace with SVM

df_after_Clust_pca <- merge(pca_df,df_Clust_1,by.x=c("State","Area.Code"),by.y=c("df_Clust$State","df_Clust$Area.Code"))



churn_model_rf = randomForest(( Churn ~ ., 
                                data = df_after_Clust_pca)
                              
                              importance(churn_model_rf)        
                              Top <- varImpPlot(churn_model_rf)   
                              
                              ### Top 15 consider variables to build the model
                              
                              Churn_model_rf1 <- churn_model_rf[,c(Top)]
                              
                              
                              
                              ### startified sample
                              library(caTools)
                              train_rows = sample.split(df_after_Clust_pca$Churn, SplitRatio=0.7)
                              train = df_after_Clust_pca[ train_rows,]
                              test  = df_after_Clust_pca[!train_rows,]
                              
                              
                              #### build logistic regression model with randomforest
                              
                              model_svm = svm( Churn ~ ., 
                                              data = train)
                              
                              
                              
                              
                              ### Predict the test observations 
                              
                              test$pred = predict( model_svm , newdata = test)
                              
                              
                              ### confusion matrix 
                              table(test$pred_class, test$Churn )
							  
							  
							  
							  
