# Planning-Inc: AdventureWorks Case Study
# Tarek Salhi


#### 1. Set up enviroment ####
setwd("C:/Users/tarek/Documents/Projects/Planning-Inc")
list.files()

#rm(list = ls()) # clean up enviroment before

install.packages("data.table")
install.packages("ggplot2")
install.packages("ROCR")
install.packages("caret")
install.packages("e1071")
library(data.table)
library(ggplot2)
library(ROCR)
library(caret)
library(e1071)


#### 2. Pre-Processing data ####
df <- as.data.frame(fread("ts_customerbase.csv", na.strings='NULL'))
#str(df)
#summary(df)
table(df$Road250_Flag)
# 0     1 
#17264  1855 => unbalanced dataset (1%)


# Remove unneeded variables
to.remove <- c("Age","Total_Spend","Total_Qty")
`%not.in%` <- Negate(`%in%`)
df <- subset(df,select = names(df) %not.in% to.remove)

# Convert categorical variables to nominal integers
df$age_band <-  as.numeric(factor(df$age_band))
df$MaritalStatus <- as.numeric(factor(df$MaritalStatus))

# Missing Values
n <- nrow(df)
missing <- as.data.frame(sapply(df,function(x) sum(is.na(x)) / n)) ##check % N/A's missing from each variable
write.csv(missing, file = "missing_data.csv", row.names=FALSE)


# Manually convert NA's with the median 
df$age_band[is.na(df$age_band)] <- median(df$age_band, na.rm = T)
df$HomeOwnerFlag[is.na(df$HomeOwnerFlag)] <- median(df$HomeOwnerFlag, na.rm=T)
df$MaritalStatus[is.na(df$MaritalStatus)] <- median(df$MaritalStatus, na.rm=T)
df$NumberCarsOwned[is.na(df$NumberCarsOwned)] <- median(df$NumberCarsOwned, na.rm=T)
df$TotalChildren[is.na(df$TotalChildren)] <- median(df$NumberCarsOwned, na.rm=T)
df$Tenure[is.na(df$Tenure)] <- round(mean(df$Tenure, na.rm=TRUE))


# Calculate correlation pairs - then remove unneeded variables
cor.matrix <- cor(na.omit(df[,-c(1,3)])) # online shopper excluded as it contains zero variance once NAs are removed

cutoff <- 0.75
names <- colnames(cor.matrix)
check <- matrix(NA, 18, 18)
k = 1
for (i in 1:ncol(cor.matrix)) 
{
  l = 1
  for (j in 1:nrow(cor.matrix))
  { 
    if ( (i!=j) & (abs(cor.matrix[i,j]) >= cutoff))  
    {
      l = l + 1
      check[k, l] = names[j]
    }
  } #j 
  if (l > 1)
  {
    check[k, 1] =  names[i]
    print(check[k,1:l]) # print correlation pairs
    k = k + 1  
  }
} #i

# Investigate any correlated variables if any



#### 3. Split data into train/test set ####
s <- 0.8 # set sample size
smp_size <- floor(s * nrow(df))

set.seed(123) # set the seed to make your partition reproductible
train_index <- sample(seq_len(nrow(df)), size = smp_size)
train <- df[train_index, ]
test <- df[-train_index, ]


##### 4. Create Logistic Regression Model #####
fit <- glm(Road250_Flag~
             Online_Shopper
           + gender
           + HomeOwnerFlag
           + MaritalStatus
           + NumberCarsOwned
           + TotalChildren
           + Tenure
           + age_band
           + Spend_Percentile
           + Qty_Percentile
              
            ,data=train
            ,family=binomial()
            ,control = list(maxit = 50)
           )

# Perform backward stepwise variable selection based on AIC
fit.back <- step(fit)
formula(fit.back)
summary(fit.back)


ps.model <- step(fit)
summary(ps.model)

# Create predictions
pred.df <- data.frame(CustomerID=test$CustomerID, Road250_Flag=test$Road250_Flag) #  store results here
pred.df$prob <- predict(fit.back, newdata=test, type="response")

# Investigate score distribution and set threshold cut-off
par(mfrow=c(1,2)) # set graphics display dimensions
hist(pred.df$prob, main = "Propensity Score - Histogram") 
plot(density(pred.df$prob), main = "Propensity Score - Density Plot")

t <- 0.1 # threshold for which probabilities will be classified 0 or 1
pred.df$pred <- ifelse(pred.df$prob > t, 1, 0)
write.csv(pred.df, file = "Road250_predictions.csv", row.names=FALSE)



##### 5. Model Accuracy #####
con.matrix <- confusionMatrix(pred.df$pred, pred.df$Road250_Flag)
con.matrix

#con.matrix$byClass
#con.matrix$table[1]

# Calculate AUC and plot ROC curve
pred <- prediction(pred.df$prob, pred.df$Road250_Flag)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")

auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

gini <- 2*auc -1 # a gini coefficient of greather than 0.6 represents a good model

roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="GLM")

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC = ", round(auc,5)))


##### 6. Deploymemt - Score entire base #####

save(fit.back, file = "my_model.rda")
#load("my_model1.rda")

score.df <- data.frame(CustomerID=df$CustomerID, Prop_Score = predict(fit.back, newdata = df, type="response"))
write.csv(score.df, file = "final_scores.csv", row.names=FALSE)
