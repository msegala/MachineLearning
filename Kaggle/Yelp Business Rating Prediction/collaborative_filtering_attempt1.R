IdLookup <- read.csv("C:/Users/segm/Desktop/MachineLearning/Kaggle/Yelp Business Rating Prediction/IdLookupTable.csv", 
                     header=TRUE, stringsAsFactors=FALSE)


# Get the list of users and buisnesses from the appropriate tables
users <- c(dat.user[,2])
buisness <- c(dat.bus[,1])
size <- length(buisness)*length(users)

# Fill a sparse matrix with the appropriate rows (users) and columns (buisnesses) 
mat <- Matrix(data = c(0),nrow = length(users), ncol = length(buisness),
              dimnames = list(Origin      = users,
                              Destination = buisness),
              sparse=TRUE)

# Loop over all the reviews from the review table and fill in the matrix

size <- nrow(dat.review)
#size<-1000
pb <- winProgressBar(title = "progress bar", min = 0,
                     max = size, width = 300)

for (i in 1:size){
  
  # Get the user, buisness, and rating in each review 
  user <- dat.review[i,2]
  buis <- dat.review[i,8]
  rating <- dat.review[i,4]
  
  #print(sprintf("User / buis / rating = %s %s %s",user,buis,rating))
  
  # Only fill the matrix if that user/buisness entry exists
  if ( (user %in% names(mat[,1])) && (buis %in% names(mat[1,])) ){
    
    #find this entry in the matrix and fill it with the rating
    mat[user,buis] <- as.numeric(rating)
    
  }
  setWinProgressBar(pb, i, title=paste( round(i/size*100, 0),
                                        "% done"))  
}
close(pb)

summary(mat)
rm(dat.review)


# Due to issues with memory, only take the first 10000 users, convert sparse martix into matrix (need this)
# for realRatingMatrix and assign all unfilled valued to NA
small_matrix <- as(mat[1:500,],"matrix")
rm(small_matrix)
small_matrix[small_matrix < 0.1] <- NA

#Convert this into a realRatingMatrix
r <- as(small_matrix, "realRatingMatrix")
#r <- as(mat, "realRatingMatrix")  
rm(r)


rec=Recommender(r[1:490,],method="UBCF", param=list(normalize = "Z-score",method="Jaccard",nn=100, minRating=1))
#rec=Recommender(r[1:10],method="IBCF", param=list(normalize = "Z-score",method="Jaccard", minRating=1))
#rec=Recommender(r[1:990],method="POPULAR", param=list(normalize = "Z-score"))
#rec=Recommender(r[1:490,],method="POPULAR", param=list(normalize = "Z-score"))
rec
rm(rec)
as(r[991,],"list") 
as(r,"list")

# create predictions
recom <- predict(rec, r[496], n=1, type="ratings")
recom
#as(recom["uR2aNW75R4oYs9w7aw-_kQ"], "list")
rm(recom)
head(as(recom, "list"))

aa = as(recom, "list")



#scheme <- evaluationScheme(r[1:9990,], method="split", train=0.9, given=1)
scheme <- evaluationScheme(r[1:9990,], method="cross-validation", given=1, goodRating=2, k=4)


r1 <- Recommender(getData(scheme, "train"), method="UBCF",
                  param=list(normalize = "Z-score",method="Cosine",nn=50, minRating=2))
#r1

r2 <- Recommender(getData(scheme, "train"), method="POPULAR", param=list(normalize = "Z-score"))
#r2

p1 <- predict(r1, getData(scheme, "known"), type="ratings")
p1


p2 <- predict(r2, getData(scheme, "known"), type="ratings")
p2
#as(p2, "list")
head(as(p2, "list"))


#calculate error between the prediction and the unknown part of the test data
error <- rbind(
  calcPredictionError(p1, getData(scheme, "unknown")),
  calcPredictionError(p2, getData(scheme, "unknown"))
)

rownames(error) <- c("UBCF","POPULAR")
error

calcPredictionError(p2, getData(scheme, "unknown"))
calcPredictionError(p1, getData(scheme, "unknown"))

