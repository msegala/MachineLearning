# makes the SVM submission
library(caret)
library(doSNOW)




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
smallTest <- test[1:10000,]
smallTrain <- train[1:10000,]  

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
xTrain    <- xTrain[,-zero_var_train]
xVal      <- xVal[,-zero_var_train]
xTrainFull <- xFullTrain[,-zero_var_train]
TestFull   <- test[,-zero_var_train]





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



#Parallize
nCores <- 2
c1 <- makeCluster(nCores, type = "SOCK")
registerDoSNOW(c1) 
clusterExport(c1, c("%do%","foreach"))

## load libraries on workers
clusterEvalQ(c1, library(caret)) 



###############################
#
#     Steps for parameter tuning
# 
#     1) Use tuneLength = 5 (for example) which will build a 5x5 grid 
#        and try different combinations. It will use sigest() and come 
#        up with a very good sigma to use.
#
#     2) Use tuneGrid with the fixed sigma value and maybe a range of costs. 
#        tuneGrid = expand.grid(.sigma=c(0.0118),.C=c(8,16,32,64,128))
#
#     3) With the optimal value of sigma and costs, use these in the training data
#        tuneGrid = expand.grid(.sigma=c(0.0118),.C=c(64)),
#


tic=proc.time()[3]

rdaGrid_Poly = data.frame(.C=1, .degree=3, .scale=1e-07)

ctrl_Poly <- trainControl(method = "repeatedcv",
                     repeats = 3,
                     classProbs = TRUE
                    )

model <- train(xVal, yValLabels, 
               method='svmPoly',
               #tuneGrid = rdaGrid_Poly,
               trControl = ctrl_Poly,
               tuneLength = 2,
               #metric = "ROC"
              )


ctrl_Radial <- trainControl(method = "cv",
                            number = 5,
                            classProbs = TRUE
                            )



# Find a good Sigma (turned out to be sigma = 2.935701ee-07)
model <- train(xVal, yValLabels, 
               method='svmRadial',
               #trControl = ctrl_Radial,
               trControl = ctrl_Poly,
               tuneLength = 5
              )


# Find a good Cost (turned out to be C = 2)
model <- train(xVal, yValLabels, 
               method='svmRadial',
               trControl = ctrl_Radial,
               tuneGrid = expand.grid(.sigma=c(2.88e-07),.C=c(1,2,4,8,16,32))
              )



# Full training example
model <- train(xTrain, yFullTrainingLabels, 
               method='svmRadial',
               tuneGrid = expand.grid(.sigma=c(2.94e-07),.C=c(2))
              )




toc=proc.time()[3] - tic
toc

print(model)
plot(model)

predY <- predict(model, xVal)
predY
table(predY, yValLabels)

confusionMatrix(predY,yValLabels)


stopCluster(c1)




tic=proc.time()[3]
rdaGrid = data.frame(.C=1, .degree=1, .scale=2.26e-07)


ctrl <- trainControl(method = "repeatedcv",
                     repeats = 2,
                     classProbs = TRUE
                    )




model <- train(xVal, yValLabels, 
               method='svmPoly',
               #tuneGrid = rdaGrid,
               trControl = ctrl,
               tuneLength = 3,
               metric = "ROC"
              )


model <- train(xVal, yValLabels, 
               method='svmPoly',
               #tuneGrid = rdaGrid,
               trControl = ctrl,
               tuneLength = 3,
               metric = "ROC"
)





model <- train(xTrain, yTrainingLabels, 
               method='svmPoly',
               tuneGrid = rdaGrid
              )


model <- train(xTrain, yTrainingLabels, 
               method='svmPoly',
               tuneLength = 1
              )


toc=proc.time()[3] - tic


print(model, printCall = FALSE)



print(model)
plot(model)

predY <- predict(model, xVal)
table(predY, yValLabels)

confusionMatrix(predY,yValLabels)



















