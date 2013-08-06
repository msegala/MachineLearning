library(RJSONIO)
library(doSNOW)
#library(doMC)
library(plyr)
library(Matrix)
library(caret)
library(recommenderlab)
library(data.table)

# Function to randomly split the data
splitdf <- function(dataframe, nSplits = 2, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)/nSplits))
  trainset <- dataframe[-trainindex, ]
  validationset <- dataframe[trainindex, ]
  list(trainset=trainset,validationset=validationset)
}


#################################################################################
#
# Parallelize the data
#
#


#############
##  ON WINDOWS
nCores <- 2
c1 <- makeCluster(nCores, type = "SOCK")
registerDoSNOW(c1) 
clusterExport(c1, c("%do%","foreach"))

## load libraries on workers
clusterEvalQ(c1, library(caret)) 

## Stop it after the job runs
stopCluster(c1)

#############
##  ON MAC
registerDoMC(cores = 64)


convertJSON <- function(f){
  dat <- scan(f,what=character(),sep="\n")
  return(do.call(rbind.fill, lapply(dat,function(x) data.frame(lapply(fromJSON(x),paste,collapse=" | "),stringsAsFactors=FALSE))))
}

#setwd("/Users/michaelsegala/Documents/Programming/MachineLearning/Kaggle/Yelp/")
setwd("C:/Users/segm/Desktop/MachineLearning/Kaggle/Yelp Business Rating Prediction/")


dat.bus <- convertJSON("yelp_training_set/yelp_training_set/yelp_training_set_business.json")
dat.checkin <- convertJSON("yelp_training_set/yelp_training_set/yelp_training_set_checkin.json")
dat.review <- convertJSON("yelp_training_set/yelp_training_set/yelp_training_set_review.json")
dat.user <- convertJSON("yelp_training_set/yelp_training_set/yelp_training_set_user.json")


test.bus <- convertJSON("yelp_test_set/yelp_test_set/yelp_test_set_business.json")
test.checkin <- convertJSON("yelp_test_set/yelp_test_set/yelp_test_set_checkin.json")
test.review <- convertJSON("yelp_test_set/yelp_test_set/yelp_test_set_review.json")
test.user <- convertJSON("yelp_test_set/yelp_test_set/yelp_test_set_user.json")


for(cat in unique(unlist(strsplit(dat.bus$categories," | ",fixed=TRUE)))){
  dat.bus[gsub(" ",".",paste("category_",cat,sep=""))] <- ifelse(grepl(cat,dat.bus$categories),1,0)
}
dat.bus$categories <- NULL

for(cat in unique(unlist(strsplit(test.bus$categories," | ",fixed=TRUE)))){
  test.bus[gsub(" ",".",paste("category_",cat,sep=""))] <- ifelse(grepl(cat,test.bus$categories),1,0)
}
test.bus$categories <- NULL


#Convert the cities to unique integers
l=unique(c(as.character(dat.bus$city)))
dat.bus$city  <- as.numeric(factor(dat.bus$city, levels=l))
test.bus$city <- as.numeric(factor(test.bus$city, levels=l))





#################################################################################
#
## Run a regression technique on the training set and use that to make predicitons
## of stars in the testing set. Many buisnessses are in the testing set so we need
## to know these. This did not get the best results, use a cluster technique instead
##
##

#Take on the relevent columns in the training set and the testing set
small_data <- dat.bus[,c(4,5,8,10,11)]
small_predict <- test.bus[,c(1,2,9,10)]

small_predict[105,1] <- 1 #### hack to deal with no city in this entry


# create the validation set by splitting the training set up into 66% / 33%
splits <- splitdf(small_data, nSplits=4, seed=808)
trainingSet <- splits$trainset
validationSet  <- splits$validationset


#Get the features and labels in the training set
yTrainingLabels <- as.double(trainingSet[,4])
xTrain <- trainingSet[,-4]
yValLabels <- as.double(trainingSet[,4])
xVal <- trainingSet[,-4]


#hack to center and scale x-vals of training set
for (i in 1:ncol(xTrain)){
  xTrain[,i] <- scale(as.numeric(xTrain[,i]), center = TRUE, scale = TRUE)
  xVal[,i]   <- scale(as.numeric(xVal[,i]), center = TRUE, scale = TRUE)
}

#hack to center and scale x-vals of testing set
for (i in 1:ncol(small_predict)){
  small_predict[,i] <- scale(as.numeric(small_predict[,i]), center = TRUE, scale = TRUE)
}



# use 10-fold CV
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE
)

# use a rf model to test parameters on the CV set
rf_cv_train <- train(xVal,yValLabels,
                     method='rf',
                     trControl = ctrl,
                     #tuneGrid=expand.grid(.mtry=c(1))
                     tuneGrid=expand.grid(.mtry=c(0.01,0.1,1,2,4))
                     #tuneLength = 1
)

rf_cv_train
plot(rf_cv_train)


rf_train <- train(xTrain,yTrainingLabels,
                  method='rf',
                  trControl = ctrl,
                  tuneGrid = expand.grid(.mtry=c(1))
)
rf_train

# use a svm model 
svm_cv_train <- train(xVal,yValLabels,
                      method='svmRadial',
                      trControl = ctrl,
                      #tuneGrid=expand.grid(.mtry=c(1))
                      #tuneGrid=expand.grid(.mtry=c(1,2,4))
                      #tuneLength = 10
                      tuneGrid = expand.grid(.sigma=c(0.892),.C=c(0.001,0.01,0.1,0.25,0.5,1))
)

svm_cv_train
plot(svm_cv_train)

svm_train <- train(xTrain,yTrainingLabels,
                   method='svmRadial',
                   trControl = ctrl,
                   tuneGrid = expand.grid(.sigma=c(0.892),.C=c(0.01))
)
svm_train


# create the predictions for the testing set
predY <- predict(svm_train$finalModel, small_predict)
#predY <- predict(rf_train$finalModel, small_predict)
predY

### add the predictions as 'star' field to test.bus
test.bus["stars"] <- NA
which(colnames(test.bus)=="stars")
test.bus$stars <- predY
test.bus





#################################################################################
##
## Run a clustering algo to group similiar bussinesses
## This turns out to help A LOT!!! 
##
## Steps:
##        1) Combine all the training and test businesses together
##        2) Cluster all the businesses into 150 cluseters. Append these clusters to the business dataframe
##        3) Get the average company rating for each cluster
##        4) Later on, given a compnay with an unknown rating but a known cluster, can get the business average
##

###### remove all but buisness categories in both training and test set

# JUST THE CATEGORIES
#cluster_buis <- dat.bus[-c(2:12)]
#cluster_test_buis <- test.bus[-c(1:5,7,8,9,10,11)]

# with lat and long (this does the best so far, with 150 clusters)
cluster_buis <- dat.bus[as.numeric(-c(2:7,9,10,12))]
cluster_test_buis <- test.bus[as.numeric(-c(1:5,7,8,11))]

# with # reviews and city ID (this did poorly with 15 clusters)
#cluster_buis <- dat.bus[-c(2:3,6:12)]
#test.bus[105,1] <- 1 #hack to fix this one error
#cluster_test_buis <- test.bus[-c(3,4,5,7:11)]


# Add the two dataframes togetger
combined <- rbind.fill(cluster_buis, cluster_test_buis)
# For new columns with no values, repalce those with 0's instead of NA's
combined[is.na(combined)] <- 0



##
## Try 2: Calculate the euclidean distance between a new point and all the cluster centroid means
##
##
# cl <- kmeans(iris[,1:4],3)
# cl
# cl$centers[1,]
# cl$cluster
# iris[,1:4]
# x <- cl$centers
# x
# y <- c(4.9,3.0,1.4,0.2)
# y
# a = dist(rbind(x, y), method = "euclidean")
# a
# which.min(as.matrix(a)[y,][1,1:3])
# 
# 


# determine how many clusters there should be 
wss <- (nrow(combined[-c(1)])-1)*sum(apply(combined[-c(1)],2,var))
for (i in 2:100) wss[i] <- sum(kmeans(combined[-c(1)], 
                                     centers=i)$withinss)

plot(1:100, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")


library(fpc)

#Hack to turn long and lat into doubles instead of strings
combined[,2] <- as.double(combined[,2])
combined[,3] <- as.double(combined[,3])


#nClusters <- 90
nClusters <- 150
#Lets try 150 to start
fit <- kmeans(combined[-c(1)], nClusters) # 150 cluster solution
fit
fit$centers
# get cluster means 
aggregate(combined,by=list(fit$cluster),FUN=mean)


# append cluster assignment, putting into column 526
cluster_location <- 526
combined <- data.frame(combined, fit$cluster)
combined[1,c(1,cluster_location)]


dfrm_cluster <- data.frame(Cluster=as.integer(), Average=as.double(), stringsAsFactors=FALSE)
for (clus in 1:nClusters){

  # can get all the buisness_id's within a given cluster (as an example 9)
  withinCluster <- c(combined[combined$fit.cluster == clus,1])

  print(sprintf("Getting average for cluster value %i",clus))

  sum <- 0
  length <- 0
  
  for (i in 1:length(withinCluster)){
    
    tmp <- withinCluster[i]
    company_len <-length(dat.bus[dat.bus$business_id==tmp,10])
    
    if (company_len > 0){
      
      sum <- sum + as.numeric(dat.bus[dat.bus$business_id==tmp,10])
      length <- length + 1
            
    }    
  }
  
  avg <- sum/length
  print(sprintf("     Average = %f",avg))
  
  # Add tmp dataframe to full dataframe
  df_tmp_clus <- data.frame(Cluster=clus, Average=avg, stringsAsFactors=FALSE)
  dfrm_cluster <- rbind(dfrm_cluster,df_tmp_clus)  

}

dfrm_cluster


#### Now, given a buisness in test.bus, find what cluster it belongs to and get the avg star rating
combined[combined$business_id == "jli7Qqj43zUnmt622tMHYg",cluster_location]

# get that average from dfrm_cluster
dfrm_cluster[dfrm_cluster$Cluster==18,2]







#### NOT USING THIS YET
# # Ward Hierarchical Clustering
# d <- dist(cluster_buis, method = "euclidean") # distance matrix
# d
# fit <- hclust(d, method="ward") 
# fit
# plot(fit) # display dendogram
# groups <- cutree(fit, k=13) # cut tree into 5 clusters
# groups
# # draw dendogram with red borders around the 5 clusters 
# rect.hclust(fit, k=13, border="red")









################################################################################################################
##
##  Method 2: Collaborative Filtering
##
##    Will use CF to predict user ratings in the case where a user does not exists but the buisness
##    exists in IdLookup. Use values in dat.review to do this.
##    
##    steps:
##          1) look at the review file and make a new file with just user_id, buis_id, and stars
##          2) this proved to be way to big, so do a small hack to see which user_id's exists in 
##             IdLookup and do CF on just those users. This may not be ideal so should extend the
##             list to be those users plus a few thousand more
##          3) Once CF is ran in Mahout, get the file back and turn all the new entries into a hash table. 
##
##
##
##        Have tried Matrix Factorization and Item-based   
##        Both actually seem to give worse results
##  
##


#Convert the user_id and buisness_id from the review file to unique integers.
#This will be fed into mahout. 
csv_review <- dat.review[,c(2,8,4)] #skim the data and reorder it
csv_review
unique_user=unique(c(as.character(csv_review$user_id)))
unique_buis=unique(c(as.character(csv_review$business_id)))

csv_review$user_id  <- as.numeric(factor(csv_review$user_id, levels=unique_user))
csv_review$business_id  <- as.numeric(factor(csv_review$business_id, levels=unique_buis))
csv_review$stars  <- as.numeric(csv_review$stars)

csv_review[1:10,]
max(csv_review)

write.table(csv_review, file="review_forMahout.csv", 
            sep=",", row.names=FALSE, col.names=FALSE)


### user_id_forItemBased is from bottom loop and is unknown users and known businesses from IdLookUp
user_id_forMF <- user_id_forItemBased
user_id_forMF$user_id  <- as.numeric(factor(user_id_forMF$user_id, levels=unique_user))

final_MF <- user_id_forMF
final_MF <- final_MF[complete.cases(final_MF),]
final_MF <- unique(final_MF)
final_MF <- as.data.frame(final_MF)
names(final_MF)[1] <- "user_id"
final_MF_forlater <- final_MF
final_MF_forlater$user_id

#pick a random 2000 more training samples do run MF on
x4 <- sample(1:max(csv_review), 2000, replace=F)
x4 <- as.data.frame(x4)
names(x4)[1] <- "user_id"

#bind the users that are in the final IdLookup and the random 2000 samples
final_MF <- rbind(final_MF,x4)

#check that it is right, should be 166 + (2000)
nrow(final_MF)


### Select all the values in csv_review which correspond to the values in final_MF
csv_revew_small <- csv_review[csv_review$user_id %in% final_MF$user_id,]
nrow(csv_revew_small)


write.table(csv_revew_small, file="uniqueValues_forMF_CF.csv", 
            sep=",", row.names=FALSE, col.names=FALSE)


### Once you have done this in mahout you will get back some text
### file with an output or new users/buisness predictions. 
### Process them as follows...


# Load the new user predicted values from mahout matrix factorization
matrix_factorization <- read.table("matrix_factorization_output.txt", header=FALSE)

# Can get the new buisness id and value
aa = as.character(matrix_factorization[1,2])
as.numeric(unlist(strsplit(aa,":", fixed = TRUE)))


#########################################################################################
#          FIRST METHOD --> Allocate the full dataframe at the start
#
#n <-  nrow(matrix_factorization) * (ncol(matrix_factorization[1,])-1)
#n <-  10000 * (ncol(matrix_factorization[1,])-1)
#df_matrix_factorization <- data.frame(user_id=character(n), 
#                                      business_id=character(n), 
#                                      stars=double(n),
#                                      stringsAsFactors=FALSE)
# then in the loop
#n_row <- c(unique_user[tmp_user_id],unique_buis[tmp_buis_id],as.numeric(stars)) 
#df_matrix_factorization[row_in_DF,] <- 
#  c(unique_user[tmp_user_id],unique_buis[tmp_buis_id],as.numeric(stars)) 
#########################################################################################

#########################################################################################
#        SECOND METHOD --> use a data.table and then do a rbindlist on every iteration
#DT <- data.table(data.frame(user_id=character(1), 
#                            business_id=character(1), 
#                            stars=double(1),
#                            stringsAsFactors=FALSE))
#DT <- rbindlist(list(DT, as.list(c(unique_user[tmp_user_id],
#                                   unique_buis[tmp_buis_id],
#                                   as.numeric(stars)) )))
#########################################################################################

# Third Method --> Use a hashtable
# HashTable[["key"]] <- value
# can access with HashTable[["key"]]


######### NOT USING THIS!!!!!!!!!
rm(HashTable)
HashTable<-new.env()

mf_size <- nrow(matrix_factorization)
mf_pb <- winProgressBar(title = "progress bar", min = 0,
                        max = mf_size, width = 300)

tic=proc.time()[3]
for (jj in 1:mf_size){
  tmp_user_id <- jj
  
  for (i in 2:ncol(matrix_factorization[1,])){
    
    aa = as.character(matrix_factorization[jj,i])
    as.numeric(unlist(strsplit(aa,":", fixed = TRUE)))
    tmp_buis_id <-  as.numeric(unlist(strsplit(aa,":", fixed = TRUE)))[1]    
    stars <- as.numeric(unlist(strsplit(aa,":", fixed = TRUE)))[2]
    
    key <- paste(unique_user[tmp_user_id], unique_buis[tmp_buis_id], sep = ":")
    HashTable[[key]] <- as.numeric(stars)
    
  }
  
  setWinProgressBar(mf_pb, jj, title=paste( round(jj/mf_size*100, 0),
                                            "% done"))  
  
}
toc=proc.time()[3] - tic
toc
close(mf_pb)




######### NOT USING THIS!!!!!!!!!

##### TRYING ITEM BASED .....
# To DO THIS GOING TO HACK THE IDLOOKUP FILE AND LOOK FOR USERS WHICH
# WE CAN FIND NEW BUISNESS RAINTGS FOR THEM IN MAHOUT 

### user_id_forItemBased COMES FROM THE LOOP AT END OF SCRIPT

#transform ratings needed for itemBased collaboration
#make them into unique user ratings and only take unique entries
user_id_forItemBased
user_id_forItemBased$user_id  <- as.numeric(factor(user_id_forItemBased$user_id, levels=unique_user))
final <- user_id_forItemBased
final <- final[complete.cases(final),]
final <- unique(final)
a<-as.data.frame(unique(final))
final

write.table(final, file="uniqueValues_forItemBased_CF.csv", 
            sep=",", row.names=FALSE, col.names=FALSE)



# Load the new user predicted values from mahout item based 
count.fields("itemBased_CF_output.txt", sep = "\t")
no_col <- max(count.fields("itemBased_CF_output.txt", sep = "\t"))
item_based_CF_table <- read.table("itemBased_CF_output.txt",sep="\t",fill=TRUE,col.names=1:no_col)


rm(HashTable_itemBased)
HashTable_itemBased<-new.env()

IB_size <- nrow(item_based_CF_table)
IB_pb <- winProgressBar(title = "progress bar", min = 0,
                        max = IB_size, width = 300)
tic=proc.time()[3]
for (jj in 1:IB_size){
  tmp_user_id <- item_based_CF_table[jj,1]
  
  for (i in 2:ncol(item_based_CF_table[jj,])){
    aa = as.character(item_based_CF_table[jj,i])
  
    as.numeric(unlist(strsplit(aa,"+", fixed = TRUE)))
    tmp_buis_id <-  as.numeric(unlist(strsplit(aa,"+", fixed = TRUE)))[1]    
    stars <- as.numeric(unlist(strsplit(aa,"+", fixed = TRUE)))[2]
    
 
    if(!is.na(as.numeric(stars))){      

      key <- paste(unique_user[tmp_user_id], unique_buis[tmp_buis_id], sep = ":")
      HashTable_itemBased[[key]] <- as.numeric(stars)
      
      #print(sprintf("Key / Value = %s -> %f",key,as.numeric(stars)))
      
    }        
  }
  
  setWinProgressBar(IB_pb, jj, title=paste( round(jj/IB_size*100, 0),
                                            "% done"))  
  
}
toc=proc.time()[3] - tic
toc
close(IB_pb)




######### USING THIS!!!!!!!!!

########## FOR MATRIX BASED.....
# smaller set
#count.fields("matrixFact_CF.txt", sep = "\t")
#no_col <- max(count.fields("matrixFact_CF.txt", sep = "\t"))
#matrix_based_CF_table <- read.table("matrixFact_CF.txt",sep="\t",fill=TRUE,col.names=1:no_col)

#count.fields("matrixFact_CF_largerSet.txt", sep = "\t")
#no_col <- max(count.fields("matrixFact_CF_largerSet.txt", sep = "\t"))
#matrix_based_CF_table <- read.table("matrixFact_CF_largerSet.txt",sep="\t",fill=TRUE,col.names=1:no_col)

count.fields("matrixFact_CF_FullSet_small.txt", sep = "\t")
no_col <- max(count.fields("matrixFact_CF_FullSet_small.txt", sep = "\t"))
matrix_based_CF_table <- read.table("matrixFact_CF_FullSet_small.txt",sep="\t",fill=TRUE,col.names=1:no_col)


rm(HashTable)
HashTable<-new.env()

IB_size <- nrow(matrix_based_CF_table)
IB_pb <- winProgressBar(title = "progress bar", min = 0,
                        max = IB_size, width = 300)
tic=proc.time()[3]
for (jj in 1:IB_size){
  tmp_user_id <- matrix_based_CF_table[jj,1]
  
  print(tmp_user_id)
  
  
  if(tmp_user_id %in% final_MF_forlater$user_id){
      
    for (i in 2:ncol(matrix_based_CF_table[jj,])){
      aa = as.character(matrix_based_CF_table[jj,i])
      
      as.numeric(unlist(strsplit(aa,"+", fixed = TRUE)))
      tmp_buis_id <-  as.numeric(unlist(strsplit(aa,"+", fixed = TRUE)))[1]    
      stars <- as.numeric(unlist(strsplit(aa,"+", fixed = TRUE)))[2]
          
      if(!is.na(as.numeric(stars))){      
        
        key <- paste(unique_user[tmp_user_id], unique_buis[tmp_buis_id], sep = ":")
        HashTable[[key]] <- as.numeric(stars)
        
        #print(sprintf("Key / Value = %s -> %f",key,as.numeric(stars)))
        
      }        
    }
  
  }
  
  setWinProgressBar(IB_pb, jj, title=paste( round(jj/IB_size*100, 0),
                                            "% done"))  
  
}
toc=proc.time()[3] - tic
toc
close(IB_pb)





################################################################################################################
##
##  Method 1: Using the user and business mean stars to get a prediction
##
##  Average rating of the user in the training set. If the user was not in the training set, the global review mean was computed.
##
##  This method gets a 1.24639
##
##  Adding in regression model of  stars ~ city + review_count + longitute + latitude get a 1.24571
##
##  Adding in clustering model by grouping known and unknown buisnesses using long + lat + categories, this gets a 1.23887
##
##  Adding a Matrix Factorization, this gets 1.24402
##
##
##  CASES:
##          1) User mean and buisness mean exists - take average of two (now using 0.6*buis + 0.4*user)
##          2) User mean exists, buisness mean does not exists - take user mean
##          3) Buisness exists, user mean does not exists - using Matrix factorization or take business mean
##          4) User and Buisess do not exists - use clusters
##


user_id_forItemBased <- data.frame(user_id=as.character(),stringsAsFactors=FALSE)

IdLookup <- read.csv("IdLookupTable.csv", 
                     header=TRUE, stringsAsFactors=FALSE)

total <- nrow(IdLookup)
#total <- 1000
PRINTME <- FALSE
dfrm <- data.frame(RecommendationId=integer(total), 
                   Stars=double(total), 
                   stringsAsFactors=FALSE)

pb <- winProgressBar(title = "progress bar", min = 0,
                     max = total, width = 300)


for (i in 1:total){
  
  avg <- 0
  
  # Decompose the three columns into usable strings
  id <- IdLookup[i,]
  user_id <- toString(as.character(id[1]))
  buis_id <- toString(as.character(id[2]))
  recm_id <- toString(as.character(id[3]))
  
  # Is there a review of this already?
  #review_len <- length(dat.review[dat.review$user_id==user_id&dat.review$business_id==buis_id,4])
  
  #check if user exists in the training set
  len <- length(dat.user[dat.user$user_id==user_id,4])
  
  # check if the company exists in the training set
  company_len <-length(dat.bus[dat.bus$business_id==buis_id,10])
  
  #check if the user has given a zero rating, if so use the buisness value
  stars_tmp <- 1
  if (len == 1){
    stars_tmp <- as.numeric(dat.user[dat.user$user_id==user_id,4])
  }  
  
  
  # If there is already a review of this, take that value. This made the score yorse
  #if (review_len > 0){
  #  avg <- dat.review[dat.review$user_id==user_id&dat.review$business_id==buis_id,4]
  #  if(PRINTME) {print(sprintf("i / user / buis / rating = %i / %s / %s / %s",i,user_id,buis_id,avg))}
  #}
  
  if (len == 1 && company_len == 1){
    # Both the user and Buisness exists
    # if TakeAvg = True, then take average, otherwise take the average user_rating
    
    takeAvg <- TRUE
    
    user_rating <-  dat.user[dat.user$user_id==user_id,4]
    buis_rating <- dat.bus[dat.bus$business_id==buis_id,10]   
    
    # if more than zero stars
    if(stars_tmp > 0){
      
      if (takeAvg){
        #avg <- (as.double(user_rating) + as.double(buis_rating))/2
        avg <- (0.4*as.double(user_rating) + 0.6*as.double(buis_rating))
      }
      else{
        avg <- user_rating
      }
      
    }
    else{
      avg <- buis_rating      
    }
    
    if(PRINTME) {print(sprintf("User / buissness stars / avg = %s / %s / %s",user_rating,buis_rating,avg))}
  }
  else if( len == 1 && company_len == 0 ){
    #User exists but buisness does not exist in the training set
    
    user_rating <-  dat.user[dat.user$user_id==user_id,4]
    # if more than zero stars
    if(stars_tmp > 0){
      avg <- user_rating
    }
    else{
      avg <- 3.766723      
    }
    
  }
  else if( len == 0 && company_len == 1 ){
    #Buisness exists but user does not exist in the training set
  
    
    #Use Collaborative Filtering
    useCF <- TRUE
    
    if (useCF){

      #user_id_forItemBased_tmp <- data.frame(user_id=user_id, stringsAsFactors=FALSE)
      #user_id_forItemBased <- rbind(user_id_forItemBased,user_id_forItemBased_tmp)
      
      
      key <- paste(user_id, buis_id, sep = ":")
      #print(sprintf("Looking for key %s",key))
      
      if (is.null(HashTable[[key]])) {
        #if (is.null(HashTable_itemBased[[key]])) {  
        avg <- dat.bus[dat.bus$business_id==buis_id,10]    
        #print("IS NULL")
      }
      else{
        print("IS NOT NULL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
        #avg <- HashTable_itemBased[[key]]
        avg <- HashTable[[key]]
        print(sprintf("FOUND key %s, with average %f",key,avg))
        
      }
      
      
      
    }
    else{
      avg <- dat.bus[dat.bus$business_id==buis_id,10]    
    }
    
    if(PRINTME) {print(sprintf("buissness stars %s",avg))}
        
    
  }
  else{
    #Neither exists in the training set
    
    useRegression <- FALSE
    useClusters   <- TRUE
    
    # Use the estimated stars from regression
    if(useRegression){
      
      len_test_buis <- length(test.bus[test.bus$business_id==buis_id,1])
      
      #Does the buisness exists in the testing set  
      if(len_test_buis > 0){
        avg <- test.bus[test.bus$business_id==buis_id,382]
        if(PRINTME) {print(sprintf("buissness from test set using regression =  %s",avg))}
      }
      else{
        avg <- 3.766723
      }
      
    }
    # Use cluster means 
    else if(useClusters){
          
      #check that buisness exists in the combined set
      length_comnbined <- length(combined[combined$business_id == buis_id,cluster_location])
      
      if(length_comnbined > 0){
      
        #### Now, given a buisness in test.bus, find what cluster it belongs to and get the avg star rating
        val <- combined[combined$business_id == buis_id,cluster_location]
        
        # get that average from dfrm_cluster
        cluster_avg <- dfrm_cluster[dfrm_cluster$Cluster==val,2]
        
        avg <- cluster_avg
      }
      else{
        avg <-3.766723
      }
      
      if(PRINTME) {print(sprintf("GETTING CLUSTER AVERAGE %s",avg))}      
      
    }
    else{
      avg <- 3.766723 
    }
    
    if(PRINTME) {print(sprintf("buissness default %s",avg))}    
    
  }
  
  # Add tmp dataframe to full dataframe
  #df_tmp <- data.frame(RecommendationId=i, Stars=avg, stringsAsFactors=FALSE)
  #dfrm <- rbind(dfrm,df_tmp)
  
  dfrm[i,] <- c(i,avg) 

  
  setWinProgressBar(pb, i, title=paste( round(i/total*100, 0),
                                        "% done"))
}
if(PRINTME) {print(dfrm)}
# write to file
write.table(dfrm, file="submission_using_MF_CollaborativeFiltering_fullSet.csv", 
            sep=",", row.names=FALSE, col.names=TRUE)

close(pb)







