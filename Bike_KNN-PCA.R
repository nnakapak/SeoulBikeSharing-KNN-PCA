library(dplyr) # Package for subset data
library(standardize)
library(class) # KNN
library(reshape2) # Heat map
library(hrbrthemes) # color for plots
library(ggplot2)
library(factoextra) # pca using prcomp

# Balance sizes of classes in excel
# Low: count < 260 (size = 2872)
# Med: 260 <= count <= 850 (size = 2942)
# High: 850 < count (size = 8760)
attach(SeoulBikeData) # including classes column (Low, Med, High)

# Data cleaning
# This data have no missing value
# All data except class column are numerical
SeoulBike_clean <- select(SeoulBikeData,-c(1,2)) # Discard first 2 columns

# Standardize each feature
SBike <- SeoulBike_clean %>% mutate_if(is.numeric, function (x) as.vector(scale(x))) # scaling by (xj-mj)/sd
mean(SBike[,3]) # mean = 1.334085e-16 (about 0) which is good
var(SBike[,3]) # sd = 1 which is good


################################# Histograms ###################################

# Create subset for each class Low, Med, High
LOW <- SBike[SBike$Classes == "LOW",]
MED <- SBike[SBike$Classes == "MED",]
HIGH <- SBike[SBike$Classes == "HIGH",]

# Histograms for hour of the day
par(mfrow = c(1,3))
hist(LOW$Hour, main = "Hour for Low Bike Count" , col = "lightblue", xlab = "Hour of the Day")
hist(MED$Hour, main = "Hour for Med Bike Count" , col = "lightblue", xlab = "Hour of the Day")
hist(HIGH$Hour, main = "Hour for High Bike Count" , col = "lightblue", xlab = "Hour of the Day")

# Histograms for temp for each class
par(mfrow = c(1,3))
hist(LOW$Temp, main = "Temperature for Low Bike Count" , col = "lightblue", xlab = "Temp")
hist(MED$Temp, main = "Temperature for Med Bike Count" , col = "lightblue", xlab = "Temp")
hist(HIGH$Temp, main = "Temperature for High Bike Count" , col = "lightblue", xlab = "Temp")

# Histograms for seasons for each class
par(mfrow = c(1,3))
hist(LOW$Seasons, main = "Seasons for Low Bike Count" , col = "lightblue", xlab = "Seasons")
hist(MED$Seasons, main = "Seasons for Med Bike Count" , col = "lightblue", xlab = "Seasons")
hist(HIGH$Seasons, main = "Seasons for High Bike Count" , col = "lightblue", xlab = "Seasons")

# Histograms for solar
par(mfrow = c(1,3))
hist(LOW$Solar, main = "Solar for Low Bike Count" , col = "lightblue", xlab = "Solar Radiation")
hist(MED$Solar, main = "Solar for Med Bike Count" , col = "lightblue", xlab = "Solar Radiation")
hist(HIGH$Solar, main = "Solar for High Bike Count" , col = "lightblue", xlab = "Solar Radition")

# Histograms for snowfall for each class
par(mfrow = c(1,3))
hist(LOW$Snowfall, main = "Snowfall for Low Bike Count" , col = "lightblue", xlab = "Snowfalld")
hist(MED$Snowfall, main = "Snowfall for Med Bike Count" , col = "lightblue", xlab = "Snowfall")
hist(HIGH$Snowfall, main = "Snowfall for High Bike Count" , col = "lightblue", xlab = "Snowfall")

################################### KS Test ####################################

# Hour
ks.test(LOW$Hour,MED$Hour)
ks.test(LOW$Hour,HIGH$Hour)
ks.test(MED$Hour,HIGH$Hour)

# Temp
ks.test(LOW$Temp,MED$Temp)
ks.test(LOW$Temp,HIGH$Temp)
ks.test(MED$Temp,HIGH$Temp)

# Seasons
ks.test(LOW$Seasons,MED$Seasons)
ks.test(LOW$Seasons,HIGH$Seasons)
ks.test(MED$Seasons,HIGH$Seasons)

# Solar
ks.test(LOW$Solar,MED$Solar)
ks.test(LOW$Solar,HIGH$Solar)
ks.test(MED$Solar,HIGH$Solar)

# Snowfall
ks.test(LOW$Snowfall,MED$Snowfall)
ks.test(LOW$Snowfall,HIGH$Snowfall)
ks.test(MED$Snowfall,HIGH$Snowfall)

###################### Creating train and test sets ############################

# Creating the 80% random train set interval by taking ONLY using LOW
LOW1 = SBike[which(SBike$Classes =="LOW"),]
n_l <- nrow(LOW1[which(LOW1$Classes =="LOW"),])
trainset_l <- sample(1:n_l, 0.8*n_l)

trainset_LOW <- LOW1[trainset_l,]
testset_LOW <- LOW1[-trainset_l,]

# Creating the 80% random train set interval by taking ONLY using MED
MED1 = SBike[which(SBike$Classes =="MED"),]
n_m <- nrow(MED1[which(MED1$Classes =="MED"),])
trainset_m <- sample(1:n_m, 0.8*n_m)

trainset_MED <- MED1[trainset_m,]
testset_MED <- MED1[-trainset_m,]

# Creating the 80% random train set interval by taking ONLY using HIGH
HIGH1 = SBike[which(SBike$Classes =="HIGH"),]
n_h <- nrow(HIGH1[which(HIGH1$Classes =="HIGH"),])
trainset_h <- sample(1:n_h, 0.8*n_h)

trainset_HIGH <- HIGH1[trainset_h,]
testset_HIGH <- HIGH1[-trainset_h,]

# Combining the sets to full trainset and testset
Training_set <- rbind(trainset_LOW,trainset_MED,trainset_HIGH)
Test_set <- rbind(testset_LOW,testset_MED,testset_HIGH)

# Train and test labels
TRAIN_Bike_no <- Training_set[,-13]
TRAIN_Bike_label <- Training_set[, "Classes"]
TEST_Bike_no <- Test_set[,-13]
TEST_Bike_label <- Test_set[,"Classes"]


############################ Plot % accuracy at each K #########################

# Fit the model on the training set for test set at each K
set.seed(1)
K.set = c(5,10,15,20,25,30,40,50,100)
knn.test.acc <- numeric(length(K.set))

for (j in 1:length(K.set)){
  knn.pred <- knn(train = TRAIN_Bike_no,
                  test=TEST_Bike_no,
                  cl=TRAIN_Bike_label,
                  k = K.set[j])
  knn.test.acc[j] <- mean(knn.pred == TEST_Bike_label)
}

# Fit model on training set which will be higher
set.seed(1)
K.set = c(5,10,15,20,25,30,40,50,100)
knn.train.acc <- numeric(length(K.set))

for (j in 1:length(K.set)){
  knn.pred <- knn(train=TRAIN_Bike_no,
                  test=TRAIN_Bike_no,
                  cl=TRAIN_Bike_label,
                  k=K.set[j])
  knn.train.acc[j] <- mean(knn.pred == TRAIN_Bike_label)
}

# Plot percent accuracy at each K
# Red = test set  Blue = training set
plot(K.set, knn.train.acc, xlab = "K", ylab = "Accuracy", main = "Accuracy vs K", type="o", col="blue", pch="o", lty=1)
points(K.set, knn.test.acc, col="red", pch="*")
lines(K.set, knn.test.acc, col="red",lty=2)


#################### Display values of % accuracy at each K ####################

# Finding percent accuracy for each value of 5,10...100 for train set
set.seed(1)
i = 1
k.optm = 1
for (i in seq(5, 100, by = 5)){
  knn.mod <- knn(train = TRAIN_Bike_no, test = TRAIN_Bike_no, cl = TRAIN_Bike_label, k = i)
  k.optm[i] <- 100 * sum(TRAIN_Bike_label == knn.mod)/ NROW(TRAIN_Bike_label)
  k=i
  cat(k,"=", k.optm[i],'\n')
}

# Finding percent accuracy for each value of 5,10...100 for test set
set.seed(1)
i = 1
k.optm = 1
for (i in seq(5, 100, by = 5)){
  knn.mod <- knn(train = TRAIN_Bike_no, test = TEST_Bike_no, cl = TRAIN_Bike_label, k = i)
  k.optm[i] <- 100 * sum(TEST_Bike_label == knn.mod)/ NROW(TEST_Bike_label)
  k=i
  cat(k,"=", k.optm[i],'\n')
}

# Best k = 5

########################## Confusion Matrix K=5 ################################

set.seed(1)
knn.predtrainbest <- knn(train = TRAIN_Bike_no,
                         test = TRAIN_Bike_no,
                         cl = TRAIN_Bike_label,
                         k = 5)
set.seed(1)
knn.predtestbest <- knn(train = TRAIN_Bike_no,
                        test = TEST_Bike_no,
                        cl = TRAIN_Bike_label,
                        k = 5)

# Displaying the percent accuracy
mean(knn.predtrainbest == TRAIN_Bike_label)
mean(knn.predtestbest == TEST_Bike_label)

# Displaying confusion matrix
train_cmatrix <- table(data.frame(knn.predtrainbest,TRAIN_Bike_label))
print(train_cmatrix)
test_cmatrix <- table(data.frame(knn.predtestbest, TEST_Bike_label))
print(test_cmatrix)


##################################### PCA ######################################

SBIKE <- scale(SBike[,-c(13)]) # No class column
Cor_Bike<-cor(SBIKE) # Find correlation

# Creating a heat map of correlation matrix
melted_Bike<-melt(Cor_Bike)
head(melted_Bike)
ggplot(data = melted_Bike,aes(x=Var1,y=Var2,fill=value)) +
  geom_tile() + scale_fill_gradient(low = "white", high = "lightblue")+theme_ipsum()

# Compute PCA
bike_pca <- prcomp(SBike[,-13], scale = TRUE) # Need to be scale

# Visualize eigenvalues (Scree Plot)
fviz_eig(bike_pca, ncp = 12, addlabels = TRUE, barfill = "lightblue") # number of dimensions = 12
bike_eig_val <- get_eigenvalue(bike_pca) # Extracted eigenvalues & proportion of variances
bike_eig_val
# 95% of the info (variances) contained in the data are retained by the 
# first nine principal components (r)

# Visualize the contribution to r 1-9
fviz_contrib(bike_pca, choice = "var", axes = 1:9, top = 12, col = "lightblue", fill = "lightblue")
# Red dashed line on the graph above indicates the expected avg contribution
# contribute the most to the dimensions 1-9 for 95%.


############################ New subset based on PCA ###########################

# Taking out 5 features solar(7), temp(2), visibility(5), humidity(3), seasons (10)
Training_set_new <- select(Training_set,-c(2,3,5,7,10))
Test_set_new <- select(Test_set,-c(2,3,5,7,10))
  
# New train and test labels
TRAIN_Bike_no_new <- Training_set_new[,-8]
TRAIN_Bike_label_new <- Training_set_new[, "Classes"]
TEST_Bike_no_new <- Test_set_new[,-8]
TEST_Bike_label_new <- Test_set_new[,"Classes"]


########################### Confusion Matrix after PCA #########################

set.seed(1)
knn.predtrainbest_new <- knn(train = TRAIN_Bike_no_new,
                             test = TRAIN_Bike_no_new,
                             cl = TRAIN_Bike_label_new,
                             k = 5)
set.seed(1)
knn.predtestbest_new <- knn(train = TRAIN_Bike_no_new,
                            test = TEST_Bike_no_new,
                            cl = TRAIN_Bike_label_new,
                            k = 5)

# Displaying the percent accuracy
mean(knn.predtrainbest_new == TRAIN_Bike_label_new)
mean(knn.predtestbest_new == TEST_Bike_label_new)
# Accuracy went down due to less features

## displaying confusion matrix
train_cmatrix_new <- table(data.frame(knn.predtrainbest_new,TRAIN_Bike_label_new))
print(train_cmatrix_new)
test_cmatrix_new <- table(data.frame(knn.predtestbest_new, TEST_Bike_label_new))
print(test_cmatrix_new)












