# makes the random forest submission
memory.limit(40000)
library(randomForest)
library(DAAG)
library(varSelRF)
library(caret)
require(graphics)


#################################################################################
#
# Setup the data
#
# We first load in the data and then just take the first 1000 example for 
# quick testing
#
# The training data is split 66% / 33% for training/CV
#

train <- read.csv("C:/Users/segm/Desktop/Kaggle/Digit Recongnizer/train.csv", header=TRUE)
test <-  read.csv("C:/Users/segm/Desktop/Kaggle/Digit Recongnizer/test.csv", header=TRUE)

## only look at first 1000 observations
smallTest <- test[1:42000,]
smallTrain <- train[1:42000,]  

# create the validation set by splitting the training set up into 66% / 33%
splits <- splitdf(smallTrain, nSplits=3, seed=808)
trainingSet <- splits$trainset
validationSet  <- splits$validationset

# take the first column of the training/validation set for y values, the rest for x values
yTrainingLabels <- as.factor(trainingSet[,1])
xTrain <- trainingSet[,-1]
yValLabels <- as.factor(validationSet[,1])
xVal <- validationSet[,-1]
yFullTrainingLabels <- as.factor(train[,1])
xFullTrain <- train[,-1]


# Remove columns with zero variance
zero_var_test  <- nearZeroVar(smallTest)
zero_var_train <- nearZeroVar(smallTrain)
smallTest <- smallTest[,-zero_var_train]  ### do we use zero_var_test here???
xTrain    <- xTrain[,-zero_var_train]
xVal      <- xVal[,-zero_var_train]

xTrainFull <- xFullTrain[,-zero_var_train]
TestFull   <- test[,-zero_var_train]

# Can try to randomly select sqrt(# features instead)
#rnd_features <- round(runif(28,1,784))
#xTrain    <- xTrain[,rnd_features]
#xVal      <- xVal[,rnd_features]






#
#
##################################################################################






#################################################################################
#
# Functions
#
# Function to randomly split the data
splitdf <- function(dataframe, nSplits = 2, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)/nSplits))
  trainset <- dataframe[-trainindex, ]
  validationset <- dataframe[trainindex, ]
  list(trainset=trainset,validationset=validationset)
}



# Funtion to compute Multi-labeled classification F1 score
F1 <- function(confusion){

  # get number of rows and colums
  rows <- nrow(confusion)
  cols <- ncol(confusion)
  
  # compute the precision, recal and F1 score. 
  #
  # prec = tp / (tp + fp)  (this is the final column in the confusion matrix)
  # rec  = tp / (tp + fn)
  # F1   = 2*prec*rec / (prec + rec)
  # 
  # For multi-labels classification we average over the F1 scores
  # (1/M) * Sum(F1)
  #
    
  F1 <- 0
  for(i in 1:rows) {
    
    tp <- confusion[i,i]
    prec <- confusion[i,cols]
    
    fn <- sum(confusion[1:rows,i]) - tp
    
    rec <- tp / (tp + fn)
    F1_tmp <- (2*prec*rec)/(prec + rec)

    F1 <- F1 + F1_tmp  
  }

  # return the F1 score
  F1/rows

}



# Funtion to compute error rate
Error_rate <- function(confusion){
  
  # compute the error rate
  #
  # error = (sum of off diagonal elements) / (total sum of elements)
    
  # get number of rows and colums
  rows <- nrow(confusion)
  cols <- ncol(confusion)
  di <- 0
  off_di <- 0
  
  for(i in 1:rows) {
    
    tmp_diagonal <- confusion[i,i]
    tmp_off_diagoanl <- sum(confusion[1:rows,i]) - tmp_diagonal
    di <- di + tmp_diagonal
    off_di <- off_di + tmp_off_diagoanl 

  }
    
  err <- off_di/(di + off_di)
  
}



# Function to test different parameters to determine which gives the highest F1 score
Test_Parameters <- function(xTrain, yTrainingLabels, xVal, yValLabels){
  
  # Create a vector of different number of trees
  #nTrees <- c(500,750,1000,1250,1500)
  nTrees <- c(100,200)
  
  length(nTrees)
    
  for(i in 1:length(nTrees)) {
    rf <- randomForest(xTrain, yTrainingLabels, xtest=xVal, ytest=yValLabels, 
                       ntree=nTrees[i], keep.forest = TRUE, importance = TRUE)
    F1_score <- F1(rf$test$confusion)
    error_rate <- Error_rate(rf$test$confusion)
    print(sprintf("i = %i, ntrees = %i : F1 = %f , error rate = %f",i,nTrees[i],F1_score,error_rate))
  }
  
}

#
#
##################################################################################




##################################################################################
#
# Begin the actual work of the program
#
#

#### Principal Components
pc = princomp(smallTrain)
summary(pc)
plot(pc,type="lines")
a=pc$scores
a[1:10,1:10]

#rf <- randomForest(xTrain, yTrainingLabels, xtest=xVal, ytest=yValLabels, ntree=1000
#                  ,keep.forest = TRUE, importance = TRUE, do.trace=300)
#rf <- randomForest(xTrain, yTrainingLabels, xtest=xVal, ytest=yValLabels, ntree=4500
#                  ,keep.forest = TRUE, importance = TRUE)

tic=proc.time()[3]
#rf <- randomForest(xTrain, yTrainingLabels, xtest=xVal, ytest=yValLabels, ntree=100
#                   ,keep.forest = TRUE, importance = TRUE)

#### RUN FULL TRAINING 
rf <- randomForest(xTrainFull, yFullTrainingLabels, xtest=TestFull, ntree=200
                   ,keep.forest = TRUE, importance = TRUE)

toc=proc.time()[3] - tic
print(sprintf("Running time: %f",toc))





rf
print(rf)
varImpPlot(rf)
varSelRF(xVal)
imp = importance(rf)
imp[1:100,]


plot(rf)
rf$oob.times

# Create prefiction on the testing set
#preds <- predict(rf, smallTest)
preds <- predict(rf, test)
preds

# Compute the F1 score using the validation confusion matrix
F1_score <- F1(rf$test$confusion)
F1_score

# Compute the error rate using the validation confusion matrix
err <- Error_rate(rf$test$confusion)
err


# Compute the F1 scores for a variety of different parameters
Test_Parameters(xTrain, yTrainingLabels, xVal, yValLabels)


#OOB.votes <- predict (rf,smallTest,type="prob");
test_prediction <- predict (rf,smallTest)
test_prediction
OOB.pred <- OOB.votes[,2];
pred.obj <- prediction (OOB.pred,yValLabels);

predictions <- levels(yTrainingLabels)[rf$test$predicted]
predictions


CV_test <- rfcv(smallTest,test_prediction)
with(CV_test, plot(n.var, error.cv, log="x", type="o", lwd=2))

CV_train <- rfcv(xTrain,yTrainingLabels)
with(CV_train, plot(n.var, error.cv, log="x", type="o", lwd=2))

CV <- rfcv(xVal,yValLabels)
CV$error.cv
with(CV, plot(n.var, error.cv, log="x", type="o", lwd=2))


# Overal the training, CV error
with(CV_train, plot(n.var, error.cv, type="o", lwd=2))
par(new=TRUE)
with(CV, plot(n.var, error.cv, type="o", lwd=2, col = "red"))
par(new=TRUE)
with(CV_test, plot(n.var, error.cv, type="o", lwd=2, col = "green"))



result <- replicate(5, rfcv(xVal,yValLabels), simplify=FALSE)
error.cv <- sapply(result, "[[", "error.cv")
error.cv
matplot(result[[1]]$n.var, cbind(rowMeans(error.cv), error.cv), type="l",
        lwd=c(2, rep(1, ncol(error.cv))), col=1, lty=1, log="x",
        xlab="Number of variables", ylab="CV Error")


tuneRF(xVal,yValLabels,10,ntreeTry=500, stepFactor=2,improve=0.5)


write(predictions, file="C:/Users/segm/Desktop/Kaggle/Digit Recongnizer/my_rf_benchmark.csv", 
      ncolumns=1) 







