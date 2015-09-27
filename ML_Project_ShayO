

install.packages('caret', dependencies = TRUE)
library(caret)
library(AppliedPredictiveModeling)
# -------- Tree
library(rpart)				        # Popular decision tree algorithm
library(rattle)					# Fancy tree plot
library(rpart.plot)				# Enhanced tree plots
library(RColorBrewer)				# Color selection for fancy tree plot
library(party)					# Alternative decision tree algorithm
#?library(partykit)				# Convert rpart object to BinaryTree

set.seed(3433)

# -------------------------------------Load Data -------------------------------------
# read training file
train <- read.csv("./pml-training.csv")

#---------------------------- explore data -------------------------------------
head(train)
summary(train)

#---------------------------- clean data ------------------------------------

#----------- remove features that seem useless

# Additional Feature Selection: 
# https://class.coursera.org/predmachlearn-031/forum/thread?thread_id=120#comment-274
# http://topepo.github.io/caret/preprocess.html
# 
# There you'll find functions for picking out features that have almost no variation 
# (nearZeroVar), or are linear combinations of each other (findLinearCombos), or are 
# very highly correlated (findCorrelations), etc.  We've already seen varImp.


# explore vert common info
temp <- (table(train$new_window)[1])
temp[1]/sum(temp) # no's are .9793. They're the same as NAs

# ---------- remove cols with almost always the same value

# table(y[1]) is the value in the first row of a table ranking appearances of a value. 
#It's the number of occurences of the mode. This line gives the % of rows having the mode.
# A very high number indicates useless data
mode_share <- as.numeric(sapply(train, function(y) (table(y)[1]/length(y))))

mode_value <-  (sapply(train, function(y) names((table(y)[1]) ) ))
#table(mode_value)

modes <- data.frame(cbind(mode_share,mode_value))#
modes$mode_share2 <- as.numeric(as.character(modes$mode_share))
head(modes,5) # note blanks are also appearing .9793. seems to be things containing yaw. maybe skewness as well

# remove features that have high mode (ie .9793)
good_cols <- modes$mode_share2<.97
bad_cols <- modes$mode_share2>.97
table(good_cols)

train_goodCols <- train[,good_cols]
#train_badCols <- train[,bad_cols]

# ----- now remove cols with almost alwaysNAs


na_count <- sapply(train_goodCols, function(y) (sum(is.na(y)))/length(y)) # for each column % of entries taht are na
#table(good_colsNA)

good_colsNA <- na_count < .97
#table(good_colsNA)

train_goodColsNA <- train_goodCols[,good_colsNA]

# # explore na count info
# head(data.frame(na_count))
# hist(na_count,breaks=seq(0,1,.01))
# table(na_count) # looks like all 0 or .979. Let's drop the non features.

# ------ remove a few more columns after inspectin

# x is just row number
# raw_timestamp_part_1 and raw_timestamep_part2 and cvtd_timestamp --> remove. might add back
# remove username. might add back

train_goodColsNA_Picked <- train_goodColsNA[,6:length(train_goodColsNA)]
train_goodColsNA_Picked_ExClasse <- train_goodColsNA_Picked[,1:(length(train_goodColsNA_Picked)-1)]

# ------------------------------------- pre process -------------------------------------

pca_train <- preProcess(train_goodColsNA_Picked_ExClasse,method="pca",thresh=.90)
trainPC <- predict(pca_train,train_goodColsNA_Picked_ExClasse) # use model to predict on same data we build model with
# perhaps glm isn't best since it's classification
modelFit <- train(train_goodColsNA_Picked$classe ~ ., method="glm", data =trainPC)

# ------------------------------------- build model -------------------------------------

# # --- Tree w single column
# 
# modRPart <- train(classe~pitch_belt,method="rpart",data=train)
# 
# print(modRPart$finalModel)
# fancyRpartPlot(modRPart$finalModel)
# 
# temp_test <- train$pitch_belt
# 
# RPart_Prediction  <- predict(modRPart,train,type="class")
# 
# TreeResults<- confusionMatrix(RPart_Prediction,train$classe)

# --- Tree w all data

ptm <- proc.time() # start clock
modRPart <- train(classe~.,method="rpart",data=train_goodColsNA_Picked)
duration <- prod.time() - ptm #stop clock

# print the tree
print(modRPart$finalModel)
fancyRpartPlot(modRPart$finalModel)

RPart_Prediction  <- predict(modRPart,train_goodColsNA_Picked,type="raw")

TreeResults<- confusionMatrix(RPart_Prediction,train$classe)
# Basic tree using all 'good' columns. Accuracy 49% 
# Note 95% CI is .488 

# treebag?

# -------- Regression
modLM <- train(classe ~ roll_dumbbell, method="lm", data=train)
pred <- predict(modLM,tbdtestdata)
qplot(classe,pred,data=tbdtestdata)

#----------- Random Forest
modRF <- train(class ~ ., method="rf", data=train)

########## ------------- Error Estimate and Validation --------- 

# Estimate the sample error with cross validation
#  You should create a report describing how you built your model, 
# how you used cross validation, what you think the expected out of sample error is, 
# and why you made the choices you did

    # inserting my model

    # define training control
    train_control <- trainControl(method="cv", number=10)
    modRPartFolds <- train(classe~.,method="rpart",data=train_goodColsNA_Picked,trControl=train_control)
    RPart_Prediction  <- predict(modRPartFolds,train_goodColsNA_Picked,type="raw")
    TreeResults<- confusionMatrix(RPart_Prediction,train_goodColsNA_Picked$classe)
    # 49%??? is this really across 10 folds?? This is just redoing the same model.
    # TODO: Is this adequate cross-validation?
        
