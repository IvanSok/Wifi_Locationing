# Libraries: --------------------------------------------------------------
pacman::p_load(RMySQL, ggpubr, ggfortify, stringr,  dplyr, lubridate, ggplot2, gdata, readr, plotly, highcharter, doParallel, parallel, reshape2, ggfortify, forecast, padr, DescTools,
               stats, xts, prophet, purrr, caret, forecast, GGally, rstudioapi, h2o, plyr, tidyr, tidyquant, fpp2, foreach, progress, randomForest)
###############################################################################
# Github setup ------------------------------------------------------------
current_path <- getActiveDocumentContext()$path

setwd(dirname(dirname(current_path)))
rm(current_path)
###############################################################################
# Importing data:
training_data <- read.csv2("datasets/trainingData.csv", header = TRUE, sep = ",",
                           stringsAsFactors = FALSE, na.strings = c("NA", "-", "?"))
validation_data <- read.csv("datasets/validationData.csv",header = TRUE, sep = ",",
                            stringsAsFactors = FALSE, na.strings = c("NA", "-", "?"))


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

master_data <- gdata::combine(training_data, validation_data)
master_data <- master_data %>% mutate(ID = row_number())  #doesn't work

factorised <- c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID",
                "PHONEID", "source")

master_data[,factorised] <- lapply(master_data[,factorised], as.factor)
rm(factorised)

numerical <- c("LONGITUDE", "LATITUDE")
master_data[,numerical] <- lapply(master_data[,numerical], as.numeric)
rm(numerical)

master_data$TIMESTAMP <- as_datetime(master_data$TIMESTAMP, origin = "1970-01-01",
                                     tz = "UTC")

#Organising all WAPS in a vector and rescaling WAP = 100 to WAP = -110
WAPS <- grep("WAP", names(master_data), value = TRUE )
master_data[, WAPS] <- sapply(master_data[, WAPS], function(x) ifelse(x==100, -110, x)) #if x = 100 then it changes to -100, if x is not equal to 100, then it just stays as x

#Removing WAPS with near zero variance:
WAPS_var_training <- nearZeroVar(master_data[master_data$source == "training_data"
                                          ,WAPS], saveMetrics = TRUE)
WAPS_var_validation <- nearZeroVar(master_data[master_data$source == "validation_data"
                                          ,WAPS], saveMetrics = TRUE)

master_data <- master_data[-which(WAPS_var_training$zeroVar == TRUE |
                                    WAPS_var_validation$zeroVar == TRUE)]


rm(WAPS_var_training, WAPS_var_validation)

# Removing rows with RSSI signal equal to -110 (no signal):
WAPS <- grep("WAP", names(master_data), value = TRUE)

master_data <- master_data %>% filter(apply(master_data[WAPS], 1, function(x) length(unique(x))) >1)

###############################################################################
#3D plot of buildings:
plot_ly(master_data, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~FLOOR, colors = c('#BF382A', '#0C4B8E'), size = 2) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))
###############################################################################
# Feature engineering:
# adding new variables with the strongest WAP and RSSI value:
master_data <- master_data %>% mutate(StrongWap = NA, StrongRSSI = NA)

master_data <- master_data %>% mutate(StrongWap = colnames(master_data[WAPS])[apply(master_data[WAPS], 1, which.max)])
master_data$StrongWap <- as.factor(master_data$StrongWap)

master_data <- master_data %>% mutate(StrongRSSI = apply(master_data[WAPS], 1, max))

# Adding BUILDINGID + FLOOR variable:
master_data$BUILDING_FLOOR <- as.factor(group_indices(master_data, BUILDINGID, FLOOR))


###############################################################################
# Train/Test sets:

master_data_split <- split(master_data, master_data$source)
list2env(master_data_split, envir = .GlobalEnv)
rm(master_data_split)

#Validation data to test performance:
data_validation <- validation_data

# Parallel Process:
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)


#Cross validation:
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,
                           allowParallel = TRUE)

###############################################################################
# Building prediction:
Building_SVM1 <- caret::train(BUILDINGID ~ StrongWap, data = training_data,
                              method = "svmLinear", trControl = fitControl)

save(Building_SVM1, file = "Building_SVM1.rda")
load("Building_SVM1.rda")

Building_pred <- predict(Building_SVM1, data_validation)
ConfusionMatrix <- confusionMatrix(Building_pred, data_validation$BUILDINGID)
ConfusionMatrix
rm(ConfusionMatrix)

# Replacing original buidling id with predicted in validation dataset:
master_data$BUILDINGID[master_data$source == "validation_data"] <- Building_pred

###############################################################################

# Predicting Longitude per building:
buildings <- split(training_data, training_data$BUILDINGID)
names(buildings) <- c("Building0", "Building1", "Building2")

list2env(buildings, envir = .GlobalEnv)

#Building0:
LONG_B0_RF <- randomForest(LONGITUDE~. -LATITUDE -FLOOR 
                           -SPACEID -RELATIVEPOSITION 
                           -USERID -PHONEID -TIMESTAMP
                           -source -StrongWap -StrongRSSI
                           -BUILDING_FLOOR, data = Building0,
                           importance = T, maximize = T, method = "rf",
                           trControl = fitControl, ntree = 100, mtry = 104,
                           allowParalel = TRUE)

save(LONG_B0_RF, file =  "LONG_B0_RF.ra")
load("LONG_B0_RF.ra")

LONG_B0_RF_Predictions <- predict(LONG_B0_RF, validation_data[validation_data$BUILDINGID == 0,])
LONG_B0_RF_postRes <- postResample(LONG_B0_RF_Predictions, data_validation$LONGITUDE[data_validation$BUILDINGID == 0])
LONG_B0_RF_postRes #performance

#Building1:
LONG_B1_RF <- randomForest(LONGITUDE~. -LATITUDE -FLOOR 
                           -SPACEID -RELATIVEPOSITION 
                           -USERID -PHONEID -TIMESTAMP
                           -source -StrongWap -StrongRSSI
                           -BUILDING_FLOOR, data = Building1,
                           importance = T, maximize = T, method = "rf",
                           trControl = fitControl, ntree = 100, mtry = 104,
                           allowParalel = TRUE)

save(LONG_B1_RF, file =  "LONG_B1_RF.ra")
load("LONG_B1_RF.ra")

LONG_B1_RF_Predictions <- predict(LONG_B1_RF, validation_data[validation_data$BUILDINGID == 1,])
LONG_B1_RF_postRes <- postResample(LONG_B1_RF_Predictions, data_validation$LONGITUDE[data_validation$BUILDINGID == 1])
LONG_B1_RF_postRes #performance

#Building2:
LONG_B2_RF <- randomForest(LONGITUDE~. -LATITUDE -FLOOR 
                           -SPACEID -RELATIVEPOSITION 
                           -USERID -PHONEID -TIMESTAMP
                           -source -StrongWap -StrongRSSI
                           -BUILDING_FLOOR, data = Building2,
                           importance = T, maximize = T, method = "rf",
                           trControl = fitControl, ntree = 100, mtry = 104,
                           allowParalel = TRUE)

save(LONG_B2_RF, file =  "LONG_B2_RF.ra")
load("LONG_B2_RF.ra")

LONG_B2_RF_Predictions <- predict(LONG_B2_RF, validation_data[validation_data$BUILDINGID == 2,])
LONG_B2_RF_postRes <- postResample(LONG_B2_RF_Predictions, data_validation$LONGITUDE[data_validation$BUILDINGID == 2])
LONG_B2_RF_postRes #performance

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
rf1_perf 

df2 <- df1
df2$BUILDINGID <- predictions_h2o[1]
df2 <- as.data.frame(df2)
###############################################################################

# Predicting the floors in each building:

buildings <- list()
buildings <- split(df2, df2$BUILDINGID) #creating a list to store separate buildings
buildings <- lapply(buildings, as.h2o)

building_number <- c(1,2)
floor_models <- list()
for (i in building_number){
  floor_models[[i]] <- h2o.randomForest(
    training_frame = buildings[[i]],
    x = 1:520,
    y="FLOOR",
    model_id = "rf_covType_v1",
    ntrees = 200,
    stopping_rounds = 2,
    score_each_iteration = T,
    seed = 1000000,
    nfolds = 5
  )
  print(floor_models[[i]]@model$cross_validation_metrics_summary)
}

models <- c(1,2,3)
floor_models[[3]] <- h2o.gbm(y = "FLOOR", x = 1:520, training_frame = buildings[[3]],
                             ntrees = 100, max_depth = 3, min_rows = 2,nfolds = 5 , seed = 123)
print(floor_models[[3]]@model$cross_validation_metrics_summary)

predictions <- list()
floor_predictions <- list()
for (j in models){
  predictions[[j]] <- h2o.predict(object = floor_models[[j]], newdata = buildings[[j]])
  floor_predictions[[j]] <- predictions[[j]][1]
}

floor_predictions <- lapply(floor_predictions, as.data.frame)
master_floor_predictions <- lapply(floor_predictions, rbind)
nrow(master_floor_predictions)

buildings <- lapply(buildings, as.data.frame)
###############################################################################
# Predicting longitude and latitude for each building:

building_id <- c(1,2,3)
lon_lat <- c("LONGITUDE", "LATITUDE")
lon_lat_models <- list()
  
for (i in building_id){
  for (j in lon_lat){
    
        ## run our first predictive model
        lon_lat_models[[i]] <- h2o.randomForest(         ## h2o.randomForest function
          training_frame = buildings[[i]],        ## the H2O frame for training
          x = 1:520,
          y = j,                          ## the target index (what we are predicting)
          model_id = "rf_covType_v1",    ## name the model in H2O
          ntrees = 200,                  
          stopping_rounds = 2,           
          score_each_iteration = T,
          nfolds = 5,
          seed = 1000000
        )                
    
        print(paste("Building", i-1, "prediction for", j))
        print(lon_lat_models[[i]]@model$cross_validation_metrics_summary)
      
    }
}  
