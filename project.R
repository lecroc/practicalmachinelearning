##### Practical Machine Learning Course Project

library(caret)
library(pgmm)
library(ggplot2)
library(randomForest)
library(data.table)
library(xgboost)
library(AppliedPredictiveModeling)
library(reshape2)

## Get Data

training<-read.csv("C:/Coursera/Practical Machine Learning/practicalmachinelearning/pml-training.csv", na.strings=c("NA", "", "#DIV/0!"))
testing<-read.csv("C:/Coursera/Practical Machine Learning/practicalmachinelearning/pml-testing.csv", na.strings=c("NA", "", "#DIV/0!"))                   

# Drop columns with NA values
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]

## Drop non-predictors

training<-training[,8:60]
testing<-testing[,8:60]

dim(training)
dim(testing)

## Center and scale predictors

trc<-training$classe
tec<-testing$problem_id

PreObj<-preProcess(training[,1:52])
t1<-predict(PreObj, training[, 1:52])
t2<-predict(PreObj, testing[, 1:52])

training<-as.data.frame(cbind(t1, classe=as.factor(trc)))
testing<-as.data.frame(cbind(t2, problem_id=tec))


## Feature Plots

# Create plots for each set of variables

# Belt
belt<-as.data.frame(cbind(training[,1:13],classe=training[,53]))
belt<-melt(belt, id="classe")
fp1<-ggplot(belt, aes(x=factor(classe), y=value, fill=factor(classe)))+geom_boxplot()+facet_grid(.~variable)+ggtitle("Value by Variable and Classe : Belt")
fp1<-fp1+ylim(-11,11)+theme(strip.text.x = element_text(size = 8, colour = "blue", angle = 90))

# Arm
arm<-as.data.frame(cbind(training[,14:26],classe=training[,53]))
arm<-melt(arm, id="classe")
fp2<-ggplot(arm, aes(x=factor(classe), y=value, fill=factor(classe)))+geom_boxplot()+facet_grid(.~variable)+ggtitle("Value by Variable and Classe : Arm")
fp2<-fp2+ylim(-11,11)+theme(strip.text.x = element_text(size = 8, colour = "blue", angle = 90))


# Dumbbell
bell<-as.data.frame(cbind(training[,27:39],classe=training[,53]))
bell<-melt(bell, id="classe")
fp3<-ggplot(bell, aes(x=factor(classe), y=value, fill=factor(classe)))+geom_boxplot()+facet_grid(.~variable)+ggtitle("Value by Variable and Classe : Dumbbell")
fp3<-fp3+ylim(-11,11)+theme(strip.text.x = element_text(size = 8, colour = "blue", angle = 90))

# Forearm
farm<-as.data.frame(cbind(training[,40:52],classe=training[,53]))  
farm<-melt(farm, id="classe")
fp4<-ggplot(farm, aes(x=factor(classe), y=value, fill=factor(classe)))+geom_boxplot()+facet_grid(.~variable)+ggtitle("Value by Variable and Classe : Forearm")
fp4<-fp4+ylim(-11,11)+theme(strip.text.x = element_text(size = 8, colour = "blue", angle = 90))


## Render plots

fp1
fp2
fp3
fp4

## Models


## Create train, test and validation sets from training data

set.seed(1234)
inBuild <- createDataPartition(y=training$classe,p=0.7, list=FALSE)
val <- training[-inBuild,]
buildData <- training[inBuild,]
inTrain <- createDataPartition(y=buildData$classe,p=0.7, list=FALSE)
extrn <- buildData[inTrain,]
extst <- buildData[-inTrain,]

## Model 1 Gradient Boosting

set.seed(2345)

fitControl<-trainControl(method="cv", number=4, search = "grid")
m1 <- train(classe ~ ., method="gbm",data=extrn)
print(m1$finalModel)
p1<-predict(m1,extst)
confusionMatrix(p1, extst$classe)


## Model 2 Random Forest

set.seed(3456)
m2 <- train(classe ~ ., method="rf",data=extrn, trControl = trainControl(method="cv"),number=4)
p2<-predict(m2,extst)
confusionMatrix(p2, extst$classe)

## Model 3 Extreme Gradient Boosting

set.seed(4567)
trdf<- data.table(extrn, keep.rownames=F)
tedf<-data.table(extst, keep.rownames = F)
fitControl<-trainControl(method="cv", number=4, search = "grid")
m3<-train(classe~., data=trdf, method="xgbLinear", trControl=fitControl)
p3<-predict(m3, tedf)
confusionMatrix(p3, extst$classe)


## Evaluate Models on Validation Set

p1val<-predict(m1, val)
cm1<-confusionMatrix(p1val, val$classe)

p2val<-predict(m2, val)
cm2<-confusionMatrix(p2val, val$classe)

p3val<-predict(m3, val)
cm3<-confusionMatrix(p3val, val$classe)


## Get model accuracy

error<-as.data.frame(as.numeric(cbind((1-cm1$overall["Accuracy"]), (1-cm2$overall["Accuracy"]), (1-cm3$overall["Accuracy"]))))
names(error)<-"ErrorRate"
Model<-c("Model 1", "Model 2", "Model 3")
error<-as.data.frame(cbind(Model, error))

## Out of sample error rate by model

EP<-ggplot(error, aes(x = factor(Model), y = ErrorRate, fill=factor(Model))) + geom_bar(stat = "identity") + geom_text(label=round(error$ErrorRate, 3))+xlab("")+ theme(legend.position="none")
EP<-EP+ggtitle("Out Of Sample Error Estimates")+theme(plot.title = element_text(hjust = 0.5))
EP

## Predict on Testing Data
FPred1<-predict(m1, testing)
FPred2<-predict(m2, testing)
Fpred3<-predict(m3, testing)

FPred1

FPred2

Fpred3
