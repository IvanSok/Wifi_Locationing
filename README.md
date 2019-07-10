# Wifi_Locationing

Project Goal: Investigate the possibility of using "WiFi Access Points" to determine a person's location inside closed facilities. Aim is to determine which machine learning models work best.

Data characteristics:
We have been provided with three datasets of observations (Wi-Fi fingerprints)for a multi-building industrial campus. Each observation is associated with a location (building, floor, and location ID). Thereby, we are using the signal intensity recorded from multiple "WiFi Access Points" within the building to determine the person's location.

Train: contains ~19K observations with 520 Wireless Access Points (WAPs), Longitude, Latitude, Floor, Timestamp, BuildingID, Relative Space ID, UserID and Phone ID

Test: contains ~1K observations with 520 Wireless Access Points (WAPs), Longitude, Latitude, Floor, Timestamp, BuildingID and Phone ID

Validation: ~5K observations with 520 WAPs, Longitude, Latitude and Floor are missing. As we have three datasets, we are allowed to merge Train and Test

Language used: R
