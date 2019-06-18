# Libraries: --------------------------------------------------------------
pacman::p_load(RMySQL, ggpubr, ggfortify, stringr,  dplyr, lubridate, ggplot2, gdata, readr, plotly, highcharter, doParallel, parallel, reshape2, ggfortify, forecast, padr, DescTools,
               stats, xts, prophet, purrr, caret, forecast, GGally, rstudioapi, h2o, plyr, tidyr, tidyquant, fpp2, foreach, progress, randomForest)
###############################################################################
# Github setup ------------------------------------------------------------
current_path <- getActiveDocumentContext()$path

setwd(dirname(dirname(current_path)))
rm(current_path)
###############################################################################
# Importing data: ------
training_data <- read.csv2("datasets/trainingData.csv", header = TRUE, sep = ",",
                           stringsAsFactors = FALSE, na.strings = c("NA", "-", "?"))
validation_data <- read.csv("datasets/validationData.csv",header = TRUE, sep = ",",
                            stringsAsFactors = FALSE, na.strings = c("NA", "-", "?"))


#attributes(training_data)#List your attributes within your data set.
#summary(training_data) #Prints the min, max, mean, median, and quartiles of each attribute.
#str(training_data) #Displays the structure of your data set.
#names(training_data) #Names your attributes within your data set.

#attributes(validation_data)#List your attributes within your data set.
#summary(validation_data) #Prints the min, max, mean, median, and quartiles of each attribute.
#str(validation_data) #Displays the structure of your data set.
#names(validation_data) #Names your attributes within your data set.

###############################################################################
# Preprocessing: -----
training_data <- distinct(training_data)
validation_data <- distinct(validation_data)

#sum(duplicated(training_data))
#sum(duplicated(validation_data))

master_data <- gdata::combine(training_data, validation_data)
#master_data <- master_data %>% mutate(ID = row_number(master_data))  #doesn't work

factorised <- c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID",
                "PHONEID", "source")

master_data[,factorised] <- lapply(master_data[,factorised], as.factor)
rm(factorised)

numerical <- c("LONGITUDE", "LATITUDE")
master_data[,numerical] <- lapply(master_data[,numerical], as.numeric)
rm(numerical)



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


#Rescaling the signal by applying exponential function:
master_data[grep("WAP", colnames(master_data))] <-
  apply(master_data[grep("WAP", colnames(master_data))],2,
        function(x) ifelse(x == -110,yes =  0,no =
                             ifelse(test = x < -92,yes =  1,no =
                                      ifelse(test = x > -21,yes =  100,no =
                                               (-0.0154*x*x)-(0.3794*x)+98.182))))




###############################################################################
# 3D plot of buildings: -----
plot_ly(master_data, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~FLOOR, colors = c('#BF382A', '#0C4B8E'), size = 2) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))
###############################################################################
# Feature engineering:----
# adding new variables with the strongest WAP and RSSI value:
master_data <- master_data %>% mutate(StrongWap = NA, StrongRSSI = NA)

master_data <- master_data %>% mutate(StrongWap = colnames(master_data[WAPS])[apply(master_data[WAPS], 1, which.max)])
master_data$StrongWap <- as.factor(master_data$StrongWap)

master_data <- master_data %>% mutate(StrongRSSI = apply(master_data[WAPS], 1, max))

# Adding BUILDINGID + FLOOR variable:
master_data$BUILDING_FLOOR <- as.factor(group_indices(master_data, BUILDINGID, FLOOR))

#Checking for the same WAPs in different buildings:
WAP_BUILDING_TEST <- master_data %>% dplyr::select(StrongWap, BUILDINGID)
WAP_BUILDING_TEST %>% dplyr::distinct() %>% group_by(StrongWap) %>% dplyr::summarise(count = n()) %>%
  filter(count>1)

#Remove strongest Waps that appear in multiple buildings:
master_data$WAP248 <- NULL #this WAP is present in 3 buildings


###############################################################################
# Train/Test sets: ----

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
# Building prediction: ----

h2o.init(
  nthreads=-1,            
  max_mem_size = "2G")    
h2o.removeAll() 

h2o_training <- as.h2o(training_data)
h2o_validation <- as.h2o(data_validation)
Building_RF_H2o <- h2o.randomForest( y = "BUILDINGID", x = "StrongWap", training_frame = h2o_training, 
                                    nfolds = 8, ntrees = 100, max_depth = 30, seed = 123)

save(Building_RF_H2o, file = "Building_RF_H2o.rda")
load("Building_RF_H2o.rda")

h2o.performance(Building_RF_H2o, h2o_validation)
Building_RF_H2o_Pred <- h2o.predict(Building_RF_H2o, h2o_validation)


# Replacing original buidling id with predicted in validation dataset:
master_data$BUILDINGID[master_data$source == "validation_data"] <- Building_RF_H2o_Pred


###############################################################################
# Predicting Longitude per building: ----
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

save(LONG_B0_RF, file =  "LONG_B0_RF.rda")
load("LONG_B0_RF.rda")

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

save(LONG_B1_RF, file =  "LONG_B1_RF.rda")
load("LONG_B1_RF.rda")

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

save(LONG_B2_RF, file =  "LONG_B2_RF.rda")
load("LONG_B2_RF.rda")

LONG_B2_RF_Predictions <- predict(LONG_B2_RF, validation_data[validation_data$BUILDINGID == 2,])
LONG_B2_RF_postRes <- postResample(LONG_B2_RF_Predictions, data_validation$LONGITUDE[data_validation$BUILDINGID == 2])
LONG_B2_RF_postRes #performance


#Adding all predictions for Longitude:
LONG_PRED_ALL <- c(LONG_B0_RF_Predictions, LONG_B1_RF_Predictions, LONG_B2_RF_Predictions)



###############################################################################
#Predicting Latitude by building:----

LAT_B0_RF <- randomForest(LATITUDE~. -LONGITUDE -FLOOR 
                           -SPACEID -RELATIVEPOSITION 
                           -USERID -PHONEID -TIMESTAMP
                           -source -StrongWap -StrongRSSI
                           -BUILDING_FLOOR, data = Building0,
                           importance = T, maximize = T, method = "rf",
                           trControl = fitControl, ntree = 100, mtry = 104,
                           allowParalel = TRUE)

save(LAT_B0_RF, file =  "LAT_B0_RF.rda")
load("LAT_B0_RF.rda")

LAT_B0_RF_Predictions <- predict(LAT_B0_RF, validation_data[validation_data$BUILDINGID == 0,])
LAT_B0_RF_postRes <- postResample(LAT_B0_RF_Predictions, data_validation$LATITUDE[data_validation$BUILDINGID == 0])
LAT_B0_RF_postRes #performance

#Building1:
LAT_B1_RF <- randomForest(LATITUDE~. -LONGITUDE -FLOOR 
                           -SPACEID -RELATIVEPOSITION 
                           -USERID -PHONEID -TIMESTAMP
                           -source -StrongWap -StrongRSSI
                           -BUILDING_FLOOR, data = Building1,
                           importance = T, maximize = T, method = "rf",
                           trControl = fitControl, ntree = 100, mtry = 104,
                           allowParalel = TRUE)

save(LAT_B1_RF, file =  "LAT_B1_RF.rda")
load("LAT_B1_RF.rda")

LAT_B1_RF_Predictions <- predict(LAT_B1_RF, validation_data[validation_data$BUILDINGID == 1,])
LAT_B1_RF_postRes <- postResample(LAT_B1_RF_Predictions, data_validation$LATITUDE[data_validation$BUILDINGID == 1])
LAT_B1_RF_postRes #performance

#Building2:
LAT_B2_RF <- randomForest(LATITUDE~. -LONGITUDE -FLOOR 
                           -SPACEID -RELATIVEPOSITION 
                           -USERID -PHONEID -TIMESTAMP
                           -source -StrongWap -StrongRSSI
                           -BUILDING_FLOOR, data = Building2,
                           importance = T, maximize = T, method = "rf",
                           trControl = fitControl, ntree = 100, mtry = 104,
                           allowParalel = TRUE)

save(LAT_B2_RF, file =  "LAT_B2_RF.rda")
load("LAT_B2_RF.rda")

LAT_B2_RF_Predictions <- predict(LAT_B2_RF, validation_data[validation_data$BUILDINGID == 2,])
LAT_B2_RF_postRes <- postResample(LAT_B2_RF_Predictions, data_validation$LATITUDE[data_validation$BUILDINGID == 2])
LAT_B2_RF_postRes #performance

LAT_PRED_ALL <- c(LAT_B0_RF_Predictions, LAT_B1_RF_Predictions, LAT_B2_RF_Predictions)

###############################################################################
# Predicting floor:----
# Add predicted longitude & latitude to DataValid
master_data$LATITUDE[master_data$source=="validation_data"] <- LONG_PRED_ALL
master_data$LONGITUDE[master_data$source=="validation_data"] <- LAT_PRED_ALL

# Split Data before modeling
master_data_split<-split(master_data, master_data$source)
list2env(master_data_split, envir=.GlobalEnv)
rm(master_data_split)


FLOOR_BUILDLATLONG_RF<-randomForest(FLOOR~. -SPACEID -RELATIVEPOSITION -USERID -PHONEID 
                                         -TIMESTAMP -source -StrongWap -StrongRSSI -BUILDING_FLOOR -BUILDINGID, 
                                         data= training_data, 
                                         importance=T,maximize=T,
                                         method="rf", trControl=fitControl,
                                         ntree=100, mtry= 34,allowParalel=TRUE)

save(FLOOR_BUILDLATLONG_RF, file = "FLOOR_BUILDLATLONG_RF.rda")
load("FLOOR_BUILDLATLONG_RF.rda")

FLOOR_PRED_RF<-predict(FLOOR_BUILDLATLONG_RF, data_validation)
FLOOR_PRED_RF_Postresamp<-postResample(FLOOR_PRED_RF, data_validation$FLOOR)
FLOOR_PRED_RF_Postresamp


###############################################################################
# Stop Parallel process: ----
stopCluster(cluster)
rm(cluster)
registerDoSEQ()

###############################################################################
# Model Performances:----

h2o.performance(Building_RF_H2o, h2o_validation)

LONG_B0_RF_postRes
LONG_B1_RF_postRes
LONG_B2_RF_postRes

LAT_B0_RF_postRes
LAT_B1_RF_postRes
LAT_B2_RF_postRes

FLOOR_PRED_RF_Postresamp

###############################################################################
#3D Plot for comparing real and predicted points: ----
final_predictions <- data.frame(LONG_PRED_ALL, LAT_PRED_ALL,  FLOOR_PRED_RF, as.data.frame(Building_RF_H2o_Pred[1]) )
names(final_predictions)[1] <- "LONGITUDE"
names(final_predictions)[2] <- "LATITUDE"
names(final_predictions)[3] <- "FLOOR"
names(final_predictions)[4] <- "BUILDINGID"
final_predictions["source"] <- "Predicted"
final_predictions$source <- as.factor(final_predictions$source)
final_predictions$FLOOR <- as.numeric(final_predictions$FLOOR)


final_real <- data_validation[c("LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID")]
final_real["source"] <- "Real"
final_real$source <- as.factor(final_real$source)
final_real$FLOOR <- as.numeric(final_real$FLOOR)

master_plot <- rbind(final_predictions, final_real)
rm(final_predictions)
rm(final_real)

plot_ly(master_plot, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, color = ~source, colors = c('#BF382A', '#0C4B8E'), size = 0.01) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Longitude'),
                      yaxis = list(title = 'Latitude'),
                      zaxis = list(title = 'Floor')))




