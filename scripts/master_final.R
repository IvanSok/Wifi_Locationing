# Libraries: --------------------------------------------------------------
pacman::p_load(RMySQL, dplyr, lubridate, ggplot2, readr, plotly, ggfortify, forecast, padr, DescTools,
               stats, xts, prophet, purrr, caret, GGally, rstudioapi, h2o, plyr)
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

# Preprocessing:
training_data <- distinct(training_data)
validation_data <- distinct(validation_data)

sum(duplicated(training_data))
sum(duplicated(validation_data))

master_data <- rbind(training_data, validation_data)

master_data$BUILDINGID <- as.factor(master_data$BUILDINGID)
master_data$FLOOR <- as.factor(master_data$FLOOR)


###############################################################################
#Plots:
plot_ly(master_data, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~FLOOR, colors = c('#BF382A', '#0C4B8E'), size = 2) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))
###############################################################################
# Train/Test sets:

trainSize<-round(nrow(master_data)*0.7) #calculating the size of training set (70%)
testSize<-nrow(master_data)-trainSize #calculating the size of testing set (30%)

training_indices<-sample(seq_len(nrow(master_data)),size =trainSize) #creating training and testing datasets
trainSet<-master_data[training_indices,]
testSet<-master_data[-training_indices,] 

###############################################################################

# MODELS FOR BUILDING PREDICTION:

# H2O Random Forest:

## Create an H2O cloud 
h2o.init(
  nthreads=-1,           
  max_mem_size = "2G")    
h2o.removeAll() 

## Load a file from disk
df1 <- as.h2o(master_data)

#Splits:
splits <- h2o.splitFrame(df1, c(0.7,0.2), seed=1234)    


train <- h2o.assign(splits[[1]], "train.hex")   
valid <- h2o.assign(splits[[2]], "valid.hex")   
test <- h2o.assign(splits[[3]], "test.hex")     


## run our first predictive model
rf1 <- h2o.randomForest(         ## h2o.randomForest function
  training_frame = train,        ## the H2O frame for training
  validation_frame = valid,      ## the H2O frame for validation (not required)
  x=1:520,                        ## the predictor columns, by column index
  y = "BUILDINGID",                          ## the target index (what we are predicting)
  model_id = "rf_covType_v1",
  keep_cross_validation_predictions = TRUE,
  ntrees = 100,                  
  stopping_rounds = 2,           
  score_each_iteration = T,      ## Predict against training and validation for
  seed = 1000000,
  nfolds = 8)                ## Set the random seed so that this can be reproduced.

summary(rf1)                     ## View information about the model.

rf1@model$validation_metrics  

predictions_h2o <- h2o.predict(object = rf1, newdata = df1)
rf1_perf <- h2o.performance(model = rf1, newdata =  df1)
rf1_perf #100% accuracy for predicting the building

df2 <- df1
df2$BUILDINGID <- predictions_h2o[1]
df2 <- as.data.frame(df2)
###############################################################################

# Predicting the floors in each building:

buildings <- list()
buildings <- split(df2, df2$BUILDINGID) #creating a list to store separate buildings
buildings <- lapply(buildings, as.h2o)

building_number <- c(1,2,3)
gbm1 <- list()
for (i in building_number){
  
 
  
  gbm1[[i]] <- h2o.gbm(y = "FLOOR", x = 1:520, training_frame = buildings[[i]], 
          ntrees = 100, max_depth = 3, min_rows = 2,nfolds = 5 , seed = 123)
  
                     ## View information about the model.
  
    
  print(paste("This is floors prediction for building", i-1))
  #predictions_gbm1 <- h2o.predict(object = gbm1, newdata = valid)
  print(gbm1)  
}

buildings <- lapply(buildings, as.data.frame)
###############################################################################
# Predicting longitude and latitude for each building:
lon_lat <- c("LONGITUDE", "LATITUDE")
for (i in building_number){
  for (j in lon_lat){
    #Splits:
    splits <- h2o.splitFrame(buildings[[i]], c(0.7,0.2), seed=1234)    
    
    train <- h2o.assign(splits[[1]], "train.hex")   
    valid <- h2o.assign(splits[[2]], "valid.hex")   
    test <- h2o.assign(splits[[3]], "test.hex")     
    
    ## run our first predictive model
    rf3 <- h2o.randomForest(         ## h2o.randomForest function
      training_frame = train,        ## the H2O frame for training
      validation_frame = valid,
      x = 1:520,
      y = j,                          ## the target index (what we are predicting)
      model_id = "rf_covType_v1",    ## name the model in H2O
      ntrees = 200,                  
      stopping_rounds = 2,           
      score_each_iteration = T,      ## Predict against training and validation for
      seed = 1000000
    )                ## Set the random seed so that this can be reproduced.
    
    #summary(rf2)                     ## View information about the model.
    
    #rf2@model$validation_metrics  
    print(paste("Building", i-1, "prediction for", j))
    predictions_h2o <- h2o.predict(object = rf3, newdata = valid)
    rf3_perf <- h2o.performance(model = rf3, newdata =  valid)
    print(rf3_perf) #100% accuracy for predicting the building 
  }
}
