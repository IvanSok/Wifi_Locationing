# Libraries: --------------------------------------------------------------
pacman::p_load(RMySQL, dplyr, lubridate, ggplot2, readr, plotly, ggfortify, forecast, padr, DescTools,
               stats, xts, prophet, purrr, caret, GGally, rstudioapi, h2o)
###############################################################################
## Create an H2O cloud 
h2o.init(
  nthreads=-1,            ## -1: use all available threads
  max_mem_size = "2G")    ## specify the memory size for the H2O cloud
h2o.removeAll() # Clean slate - just in case the cluster was already running
###############################################################################
# Github setup ------------------------------------------------------------
current_path <- getActiveDocumentContext()$path
setwd(dirname(dirname(current_path)))
rm(current_path)
###############################################################################
# Importing data:
training_data <- read.csv("datasets/trainingData.csv")
validation_data <- read.csv("datasets/validationData.csv")

attributes(training_data)#List your attributes within your data set.
summary(training_data) #Prints the min, max, mean, median, and quartiles of each attribute.
str(training_data) #Displays the structure of your data set.
names(training_data) #Names your attributes within your data set.

attributes(validation_data)#List your attributes within your data set.
summary(validation_data) #Prints the min, max, mean, median, and quartiles of each attribute.
str(validation_data) #Displays the structure of your data set.
names(validation_data) #Names your attributes within your data set.

###############################################################################
important_data <- c("FLOOR", "BUILDINGID", "SPACEID")
###############################################################################
# Preprocessing:
factorised <- c("FLOOR", "BUILDINGID", "SPACEID", "USERID", "PHONEID")
for (i in factorised){
  training_data[i] <- as.factor(training_data[i])
  validation_data[i] <- as.factor(validation_data[i])
}

###############################################################################
#Plots:
plot_ly(training_data, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~FLOOR, colors = c('#BF382A', '#0C4B8E'), size = 2) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))
###############################################################################
# Train/Test sets:

trainSize<-round(nrow(training_data)*0.7) #calculating the size of training set (70%)
testSize<-nrow(training_data)-trainSize #calculating the size of testing set (30%)

training_indices<-sample(seq_len(nrow(training_data)),size =trainSize) #creating training and testing datasets
trainSet<-training_data[training_indices,]
testSet<-training_data[-training_indices,] 

###############################################################################
# Models:
models <- c("lm", "svmLinear", "rf", "knn")
models <- c( "rf")

comb_metric <- c()

for (i in models){
  fit <- train(LONGITUDE~., data = trainSet, method = i)
  pred <- predict(fit, testSet)
  metric <- postResample(pred, testSet$LONGITUDE)
  comb_metric <- cbind(comb_metric, metric)
}

colnames(comb_metric) <- c(models)
names(comb_metric)
melted_data <- reshape::melt(comb_metric)
melted_data

ggplot(data = melted_data, aes(x = X2, y = value)) + geom_col() + facet_grid(X1~., scales = "free")

###############################################################################
## Default method:
# Random Forest:

# Utility function that converts regular data.frame into H2OFrame.
as_h2o <- function(df) {
  # Convert character columns into factor.
  for (colname in colnames(df)) {
    if (class(df[[colname]]) == "character") {
      df[[colname]] <- as.factor(df[[colname]])
    }
  }
  df <- as.h2o(df)
  df
}

# Function that creates H2O RandomForest model.
build_h2o_rf_model <- function(formula, data, ntrees = 5) {
  training_data <- data
  # variable to predict
  lhs_cols <- all.vars(lazyeval::f_lhs(formula))
  # predictors
  rhs_cols <- all.vars(lazyeval::f_rhs(formula))
  if (rhs_cols == ".") {
    # when . is specified, predict with all columns.
    rhs_cols <- colnames(training_data)[colnames(training_data) != lhs_cols]
  }
  # convert training data into H2OFrame.
  training_data <- as_h2o(training_data)
  # Train RandomForest model with H2O.
  md <- h2o.randomForest(x = rhs_cols, y = lhs_cols, training_frame = training_data, ntrees = ntrees)
  # Return model and formula as one object.
  ret <- list(model = md, formula = formula)
  class(ret) <- c("h2o_rf_model")
  ret
  }