pacman::p_load(RMySQL, ggpubr, ggfortify, stringr,  dplyr, lubridate, ggplot2, gdata, readr, plotly, highcharter, doParallel, parallel, reshape2, ggfortify, forecast, padr, DescTools,
               stats, xts, prophet, purrr, caret, forecast, GGally, rstudioapi, h2o, plyr, tidyr, tidyquant, fpp2, foreach, progress, randomForest)


test_data <- read.csv("datasets/testData.csv")

######################################################################
#Preprocessing:

test_data <- distinct(test_data)


WAPS <- grep("WAP", names(test_data), value = TRUE )
test_data[, WAPS] <- sapply(test_data[, WAPS], function(x) ifelse(x==100, -110, x)) #if x = 100 then it changes to -100, if x is not equal to 100, then it just stays as x

#Removing WAPS with near zero variance:
WAPS_var <- nearZeroVar(test_data[,WAPS], saveMetrics = TRUE)
test_data <- test_data[-which(WAPS_var$zeroVar == TRUE )]


# Removing rows with RSSI signal equal to -110 (no signal):
WAPS <- grep("WAP", names(test_data), value = TRUE)

test_data <- test_data %>% filter(apply(test_data[WAPS], 1, function(x) length(unique(x))) >1)


#Rescaling the signal by applying exponential function:
test_data[grep("WAP", colnames(test_data))] <-
  apply(test_data[grep("WAP", colnames(test_data))],2,
        function(x) ifelse(x == -110,yes =  0,no =
                             ifelse(test = x < -92,yes =  1,no =
                                      ifelse(test = x > -21,yes =  100,no =
                                               (-0.0154*x*x)-(0.3794*x)+98.182))))


######################################################################
# Feature engineering:----
# adding new variables with the strongest WAP and RSSI value:
test_data <- test_data %>% mutate(StrongWap = NA, StrongRSSI = NA)

test_data <- test_data %>% mutate(StrongWap = colnames(test_data[WAPS])[apply(test_data[WAPS], 1, which.max)])
test_data$StrongWap <- as.factor(test_data$StrongWap)

test_data <- test_data %>% mutate(StrongRSSI = apply(test_data[WAPS], 1, max))



#Checking for the same WAPs in different buildings:
#WAP_BUILDING_TEST <- master_data %>% dplyr::select(StrongWap, BUILDINGID)
#WAP_BUILDING_TEST %>% dplyr::distinct() %>% group_by(StrongWap) %>% dplyr::summarise(count = n()) %>%
#  filter(count>1)

#Remove strongest Waps that appear in multiple buildings:
#master_data$WAP248 <- NULL #this WAP is present in 3 buildings

######################################################################
h2o.init(
  nthreads=-1,            
  max_mem_size = "2G")    
h2o.removeAll() 

test_data_h2o <- as.h2o(test_data)
load("Building_RF_H2o.rda")
Building_RF_H2o_Pred <- h2o.predict(Building_RF_H2o, test_data_h2o )
h2o.performance(Building_RF_H2o, test_data_h2o)

h2o.shutdown()
######################################################################

load("LONG_B0_RF.rda")
LONG_B0_RF_postRes <- postResample(LONG_B0_RF_Predictions, data_validation$LONGITUDE[data_validation$BUILDINGID == 0])
LONG_B0_RF_Predictions <- predict(LONG_B0_RF, validation_data[validation_data$BUILDINGID == 0,])

load("LONG_B1_RF.rda")
LONG_B1_RF_Predictions <- predict(LONG_B1_RF, validation_data[validation_data$BUILDINGID == 1,])
LONG_B1_RF_postRes <- postResample(LONG_B1_RF_Predictions, data_validation$LONGITUDE[data_validation$BUILDINGID == 1])

load("LONG_B2_RF.rda")
LONG_B2_RF_Predictions <- predict(LONG_B2_RF, validation_data[validation_data$BUILDINGID == 1,])
LONG_B2_RF_postRes <- postResample(LONG_B2_RF_Predictions, data_validation$LONGITUDE[data_validation$BUILDINGID == 1])

######################################################################

load("LAT_B0_RF.rda")
LAT_B0_RF_Predictions <- predict(LAT_B0_RF, validation_data[validation_data$BUILDINGID == 0,])
LAT_B0_RF_postRes <- postResample(LAT_B0_RF_Predictions, data_validation$LATITUDE[data_validation$BUILDINGID == 0])

load("LAT_B1_RF.rda")
LAT_B1_RF_Predictions <- predict(LAT_B1_RF, validation_data[validation_data$BUILDINGID == 1,])
LAT_B1_RF_postRes <- postResample(LAT_B1_RF_Predictions, data_validation$LATITUDE[data_validation$BUILDINGID == 1])

load("LAT_B2_RF.rda")
LAT_B2_RF_Predictions <- predict(LAT_B2_RF, validation_data[validation_data$BUILDINGID == 2,])
LAT_B2_RF_postRes <- postResample(LAT_B2_RF_Predictions, data_validation$LATITUDE[data_validation$BUILDINGID == 2])

######################################################################

load("FLOOR_BUILDLATLONG_RF.rda")
FLOOR_PRED_RF<-predict(FLOOR_BUILDLATLONG_RF, data_validation)
FLOOR_PRED_RF_Postresamp<-postResample(FLOOR_PRED_RF, data_validation$FLOOR)

