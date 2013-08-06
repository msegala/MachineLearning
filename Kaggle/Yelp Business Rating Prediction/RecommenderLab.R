#### Tutorial on recommended lab....

library("recommenderlab")

#For this example we create a small arti???cial data set as a matrix
m <- matrix(sample(c(as.numeric(0:5), NA), 5000,  
             replace=TRUE, prob=c(rep(.9/6,6),.6)), ncol=10,  
             dimnames=list(user=paste("u", 1:500, sep=''),  
             item=paste("i", 1:10, sep=''))) 

m

#Convert this into a realRatingMatrix
r <- as(m, "realRatingMatrix")  
r

#view r in different ways 
as(r, "list")
as(r, "matrix")

#turn into dataframe
head(as(r, "data.frame"))

#normalize the rating matrix
r_m <- normalize(r)
r_m
as(r_m, "list")
image(r, main = "Raw Ratings")
image(r_m, main = "Normalized Ratings")



#can turn the matrix into a 0-1 binary matrix
r_b <- binarize(r, minRating=1)
as(r_b, "matrix")


#create a recommender
#rec=Recommender(r[1:4],method="POPULAR")
rec=Recommender(r[1:400],method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
rec=Recommender(r[1:400],method="UBCF", param=list(normalize = "Z-score",method="Jaccard",nn=5, minRating=1))
rec=Recommender(r[1:100],method="IBCF", param=list(normalize = "Z-score",method="Jaccard",minRating=1))
print(rec)

names(getModel(rec))
getModel(rec)$nn

# create predictions
recom <- predict(rec, r[1:100], type="ratings")

recom
as(recom, "list")
m[401]



as(r[5],"list")
as(recom, "matrix")[,1:10]



#########
#  MAKE SOME EVALUATION PREDICTIONS
#
#scheme <- evaluationScheme(r[1:490], method="split", train = .9, given=1, goodRating=2)
scheme <- evaluationScheme(r[1:490,], method="bootstrap", given=2, goodRating=2)
#scheme <- evaluationScheme(r[1:490,], method="cross-validation", given=2, goodRating=2, k=4)
#scheme <- evaluationScheme(r[1:490,], method="split", train=0.9, given=2)


r1 <- Recommender(getData(scheme, "train"), method="UBCF",
                  param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=3))
r1

r2 <- Recommender(getData(scheme, "train"), method="POPULAR")
r2

#### THIS DOES NOT WORK....YET
#r3 <- Recommender(getData(scheme, "train"), method="IBCF")
#r3


#compute predicted ratings for the known part of the test data (15 items for each user)
p1 <- predict(r1, getData(scheme, "known"), type="ratings")
p1

p2 <- predict(r2, getData(scheme, "known"), type="ratings")
p2
as(p2, "list")


#p3 <- predict(r3, getData(scheme, "known"), type="ratings")
#p3
#as(p3, "list")


#calculate error between the prediction and the unknown part of the test data
error <- rbind(
  calcPredictionError(p1, getData(scheme, "unknown")),
  calcPredictionError(p2, getData(scheme, "unknown"))
)


rownames(error) <- c("UBCF","POPULAR")
error
























#############################################################
#
#  MORE DETAILED EXAMPLE
#
#############################################################

#load jester dataset
data(Jester5k)
Jester5k

#take 1000 of users
r <- sample(Jester5k, 1000)

as(r, "matrix")

# how many non NA rows are there for the first entry
rowCounts(r[1,])

as(r[1,], "list")

# what is the average of the row
rowMeans(r[1,])


#look at the average rating for the users
hist(getRatings(r), breaks=100)
hist(getRatings(normalize(r)), breaks=100)
hist(getRatings(normalize(r, method="Z-score")), breaks=100)

#look at how many jokes each user has rated and the mean rating for each joke
hist(rowCounts(r), breaks=50)
hist(colMeans(r), breaks=20)

#query avaliable recommendatino methods for real-valued ratings data
recommenderRegistry$get_entries(dataType = "realRatingMatrix")

r <- Recommender(Jester5k[1:1000], method = "POPULAR")
r

names(getModel(r))
getModel(r)$topN

#make recommendations for two users
recom <- predict(r, Jester5k[1001:1002], n=100)
recom
as(recom, "list")

#get top 3
recom3 <- bestN(recom, n = 3)
as(recom3, "list")

#make predictions for the unrated entries 
recom <- predict(r, Jester5k[1001:1002], type="ratings")
recom11 <- predict(r, Jester5k[1001:1002])
recom
as(recom,"matrix")[,1:10]
as(recom,"list")
as(recom11,"list")





#### Evaluatino of predicted ratings
# take first 1000 user, 90% for training, 10% for test. 
e <- evaluationScheme(Jester5k[1:10], method="split", train=0.9, given=15)
e

#create user-based and itme-based collaborative filtering
r1 <- Recommender(getData(e, "train"), "UBCF")
r1

r2 <- Recommender(getData(e, "train"), "IBCF")
r2

#compute predicted ratings for the known part of the test data (15 items for each user)
p1 <- predict(r1, getData(e, "known"), type="ratings")
p1
as(getData(e, "known"), "matrix")
as(Jester5k[1:10], "matrix")
as(getData(e, "train"), "matrix")
as(p1, "list")


p2 <- predict(r2, getData(e, "known"), type="ratings")
p2
as(p2, "matrix")

#calculate error between the prediction and the unknown part of the test data
error <- rbind(
  calcPredictionError(p1, getData(e, "unknown")),
  calcPredictionError(p2, getData(e, "unknown"))
)

rownames(error) <- c("UBCF","IBCF")
error



####  Evaluation of a top-N recommender algorithm

#For this example we create a 4-fold cross validation scheme with the the Given-3 protocol,
#i.e., for the test users all but three randomly selected items are withheld for evaluation.
scheme <- evaluationScheme(Jester5k[1:1000], method="cross", k=4, given=3,goodRating=5)
scheme

results <- evaluate(scheme, method="POPULAR", n=c(1,3,5,10,15,20))
results

#get the 1st confusion matrix
getConfusionMatrix(results)[[1]]

#get the avg of all the runs
avg(results)

# plot true positive rate vs false positive rate
plot(results, annotate=TRUE)

#create precision recall plot
plot(results, "prec/rec", annotate=TRUE)



####    Comparing recommender algorithms

scheme <- evaluationScheme(Jester5k[1:1000], method="split", train = .9, k=1, given=20, goodRating=5)
scheme

algorithms <- list(
  "random items" = list(name="RANDOM", param=NULL),
  "popular items" = list(name="POPULAR", param=NULL),
  "user-based CF" = list(name="UBCF", param=list(method="Cosine",nn=50, minRating=5))
)
 

  
results <- evaluate(scheme, algorithms, n=c(1, 3, 5, 10, 15, 20))
results  
names(results)  
results[["user-based CF"]]
  
#plot the performance of the three algos
plot(results, annotate=c(1,3), legend="topleft")  
plot(results, "prec/rec", annotate=3)  
  
  
  
  
  
  
  
######## MOVIE DATASET
data(MovieLense)
head(MovieLense)
as(MovieLense,"matrix")  
  
scheme <- evaluationScheme(MovieLense, method = "split", train = .9,
                           k = 1, given = 10, goodRating = 4)

scheme  
  
algorithms <- list(
  "random items" = list(name="RANDOM", param=list(normalize = "Z-score")),
  "popular items" = list(name="POPULAR", param=list(normalize = "Z-score")),
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score",
                                                 method="Cosine",
                                                 nn=50, minRating=3)),
  "item-based CF" = list(name="IBCF", param=list(normalize = "Z-score"))
  
)

# run algorithms, predict next n movies
results <- evaluate(scheme, algorithms, n=c(1, 3, 5, 10, 15, 20))


plot(results, annotate = 1:4, legend="topleft")
plot(results, "prec/rec", annotate=3)

  
  



