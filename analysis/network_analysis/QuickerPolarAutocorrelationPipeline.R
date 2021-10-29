library(useful)
library(dplyr)
library(data.table)
setwd("~/Downloads/24k_pipeline_run_greedy/24k_pipeline_run_greedy/")
OH <- read.csv('20211018_24k_pipeline_run_aruco_data_with_track_numbers.csv', header=TRUE)
OH <- OH[complete.cases(OH[ , 3:4]),]
OH <- OH %>% filter(Tag!=-1) 
OHEdited<- OH[,1:4] 
BroodX <- 2300
BroodY <- 800
OHEdited$cX <- OHEdited$cX - BroodX
OHEdited$cY <- OHEdited$cY - BroodX
OHEditedPolar <- cbind(OHEdited, cart2pol(OHEdited$cX, OHEdited$cY, degrees=TRUE))
Freqs <- as.data.frame(table(OHEditedPolar$Tag)) %>% filter(Freq>0.5*max(OHEditedPolar$Frame))
for(i in Freqs$Var1){
  print(i)
  OH13 <- OHEditedPolar %>% filter(Tag==i)
  OH13dt = as.data.table(OH13)
  setkey(OH13dt, Frame)
  OH13Complete <- OH13dt[CJ(unique(Frame))]
  acf(OH13Complete$theta, lag=5000, pl=TRUE, na.action = na.pass)
}
