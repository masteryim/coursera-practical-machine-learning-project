---
title: "Practical Machine Learning Course Project"
author: "yhyim"
date: "2017년 6월 16일"
output: html_document
---



## 1. Introduction


### Background
Using devices such as *Jawbone Up, Nike FuelBand*, and *Fitbit* it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify *how well they do* it.

### Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. 


### Goal
The goal of this project is to predict the manner in which they did the exercise. This is the **classe** variable in the training set. To build model, we used cross validation, examined the expected out of sample error. Finally we used prediction model to predict 20 different test cases.


```r
# Load related packages
library(lattice)
library(ggplot2)
library(caret)
library(randomForest)
```

## 2. Getting and Cleaning Data


```r
# Read data
training  <- read.csv("pml-training.csv", header = TRUE, na.strings = c("NA", "", "#DIV/0!"))
testing  <- read.csv("pml-testing.csv", header = TRUE, na.strings = c("NA", "", "#DIV/0!"))
```

To estimate the out-of-sample error, I randomly split the full training data (training) into a smaller training set (subTraining) and a validation set (valTraining):

```r
set.seed(1000)
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
subTraining <- training[inTrain,]
valTraining <- training[-inTrain,]
```

Reducing the number of features by removing variables with nearly zero variance, variables that are almost always NA, and variables that don’t make intuitive sense for prediction. 

```r
# remove variables with nearly zero variance
nzv <- nearZeroVar(subTraining)
subTraining <- subTraining[, -nzv]
valTraining <- valTraining[, -nzv]

# remove variables that are almost always NA
mostlyNA <- sapply(subTraining, function(x) mean(is.na(x))) > 0.95
subTraining <- subTraining[, mostlyNA == FALSE]
valTraining <- valTraining[, mostlyNA == FALSE]

# remove variables that don't make intuitive sense for prediction, which happen to be the first five variables 
# (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp)
subTraining <- subTraining[, -(1:5)]
valTraining <- valTraining[, -(1:5)]
```

## 3. Building Model and evaluation


```r
# Use Random Forest to build model
fit <- randomForest(classe ~ ., data=subTraining)

# print final model to see tuning parameters
fit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = subTraining) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.29%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3906    0    0    0    0 0.000000000
## B    5 2653    0    0    0 0.001881114
## C    0   12 2384    0    0 0.005008347
## D    0    0   14 2235    3 0.007548845
## E    0    0    0    6 2519 0.002376238
```

```r
# use model to predict classe in validation set (valTraining)
preds <- predict(fit, valTraining, type = "class")

# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(valTraining$classe, preds)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    2 1137    0    0    0
##          C    0    5 1021    0    0
##          D    0    0    6  958    0
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9971          
##                  95% CI : (0.9954, 0.9983)
##     No Information Rate : 0.2848          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9963          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9956   0.9942   0.9958   1.0000
## Specificity            1.0000   0.9996   0.9990   0.9988   0.9992
## Pos Pred Value         1.0000   0.9982   0.9951   0.9938   0.9963
## Neg Pred Value         0.9995   0.9989   0.9988   0.9992   1.0000
## Prevalence             0.2848   0.1941   0.1745   0.1635   0.1832
## Detection Rate         0.2845   0.1932   0.1735   0.1628   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9994   0.9976   0.9966   0.9973   0.9996
```

`The accuracy is 99.7%`, thus predicted accuracy for the `out-of-sample error is 0.3%`.

This is a good result, so rather than trying additional algorithms, We will use Random Forests to predict on the test set.


## 4. Training with selected model
Before predicting on the test set, in order to produce the most accurate predictions, We repeated emoving variables with nearly zero variance on `training` and `testing` data

```r
# remove variables with nearly zero variance
nzv <- nearZeroVar(training)
training <- training[, -nzv]
testing <- testing[, -nzv]


# remove variables that are almost always NA
mostlyNA <- sapply(training, function(x) mean(is.na(x))) > 0.95
training <- training[, mostlyNA == FALSE]
testing <- testing[, mostlyNA == FALSE]

# remove variables that don't make intuitive sense for prediction, which happen to be the first five variables 
# (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp)
training <- training[, -(1:5)]
testing <- testing[, -(1:5)]

fit <- randomForest(classe ~ ., data=training)
```

## 5. Predicting Results on the Test Data

Finally, Using the model fit on `training` to predict the label for the observations in `testing`, and write those predictions to individual files

```r
final_predict <- predict(fit, testing, type = "class")
```



```r
pml_write_files = function(x) {
  for (i in 1:length(x)) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE,row.names=FALSE, col.names=FALSE)
  }
}

pml_write_files(final_predict)
```
