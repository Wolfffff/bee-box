setwd("~/Downloads")
# Load Directories --------------------------------------------------------
library(igraph)
library(tidygraph)
library(ggraph)
library(tidyr)
library(dplyr)
library(readr)
library(imputeTS)
library(xfun)
library(tnet)
library(purrr) 
#These are the required directories -- 
#most are required for tidygraph, 
#then imputeTS to impute NaNs from time series data and
#tnet, and purrr make managing dataframes easier
# Individual Frame Analysis ----------------------------------------
OH <- read.csv('crop_matching_pipeline_tests_aruco_data_with_track_numbers.csv', header=TRUE)
OH <- OH[complete.cases(OH[ , 3:4]),]
OHNEdited<- OH[,1:4] 
#These lines read the data and remove all rows missing an x/y coordinate.  There it
#takes the frame, tag, and coordinates.  

write.csv(OHNEdited, "OneHourEdited.csv", row.names = F) #ExportToRunAndrewsJob


OHN <- read.csv('OneHourNetwork.csv', header=TRUE) #Reads Andrew's Export
colnames(OHN) <- c('to', 'from', 'weight')
OHN <- OHN %>% filter(weight>0.1) #Only interacting 10% of the time or more
edges_g <- as_tbl_graph(OHN)
nodes = edges_g %>% 
  activate(nodes) %>% 
  as_tibble() %>% 
  mutate(node_number=row_number()) %>%
  print(n=10)
nodes$Xav <- 0
nodes$Yav <- 0
nodes$Cent <- 0
meanX<-mean(OHNEdited$cX) 
meanY<-mean(OHNEdited$cY) 
for(i in 1:nrow(nodes)){
  t <- nodes$name[i] 
  nodes$Xav[i] <- mean(subset(OHNEdited, Tag==t)$cX)
  nodes$Yav[i] <- mean(subset(OHNEdited, Tag==t)$cY)
  nodes$Cent[i] <- sqrt((nodes$Xav[i] - meanX)^2 + (nodes$Yav[i] - meanY)^2) #Base Measures of Centrality
}
edges = edges_g %>% 
  activate(edges) %>% 
  as_tibble() %>% print(n=10)
bee_tidy <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE) #Turn into graph format
ggraph(bee_tidy, layout = "graphopt") + 
  geom_node_point() +
  geom_edge_link(aes(width = weight), alpha = 0.3) + 
  scale_edge_width(range = c(0.2, 2)) +
  geom_node_text(aes(label = name, color=Cent), repel = TRUE) +
  labs(edge_width = "weight") +
  theme_graph() #These steps all just turn it to a graph
degreecentOH <- as.data.frame(degree_w(edges))
degreecentOH$name <- nodes$name

#"Cut Deck" Theoretical Data
OHS <- read.csv('crop_matching_pipeline_tests_aruco_data_with_track_numbers.csv', header=TRUE)
OHS <- OHS[complete.cases(OHS[ , 3:4]),] #Remove missing data
OHSEdited<- OHS[,1:4]
Mn <- min(OHSEdited$Frame)
Mx <- max(OHSEdited$Frame)
MxTag <- max(OHSEdited$Tag)
Rg <- Mx - Mn 
for(j in 1:100){
  Offsets <- sample(Rg, MxTag+1, replace=TRUE) #Generate a random offset for each individual
  OHSEdited$i <- OHSEdited$Tag + 1
  OHSEdited$OffSetFrame <- OHSEdited$Frame + Offsets[OHSEdited$i] #Move forward by the offset
  OHSEdited[OHSEdited$OffSetFrame>Mx,]$OffSetFrame <- OHSEdited[OHSEdited$OffSetFrame>Mx,]$OffSetFrame - Rg #Shift back any that goes outside the range
  OHSExport = subset(OHSEdited, select=c(OffSetFrame, Tag, cX, cY))
  colnames(OHSExport) <- c("Frame", "Tag", "cX", "cY")
  OHSExport <- OHSExport[order(OHSExport$Frame),]
  write.table(OHSExport, sep=",", file = paste0("CDD/CutDeckData",".",deparse(j), ".tsv"),row.names = FALSE)
}
setwd("~/Downloads/CDDN")
temp = list.files(pattern="*DeckData*")
myfiles = lapply(temp, read.csv,header=FALSE) #Loads all theoretical networks
Data <- myfiles %>% reduce(left_join, by = c("V1","V2"))
colnames(Data) <- c("to","from",c(1:100))
Data$mean <- rowMeans(Data[,3:102], na.rm=T)

for(i in 101:101){
  OHCut <- Data %>% select(1,2,i+2)
  colnames(OHCut) <- c('to', 'from', 'weight')
  OHCut <- OHCut[complete.cases(OHCut),]
  OHCut <- OHCut %>% filter(weight>0.1)
  edges_g <- as_tbl_graph(OHCut)
  nodes = edges_g %>% 
    activate(nodes) %>% 
    as_tibble() %>% 
    mutate(node_number=row_number()) %>%
    print(n=10)
  edges = edges_g %>% 
    activate(edges) %>% 
    as_tibble() %>% print(n=10)
  bee_tidy <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE) #Turn into graph format
  G <- ggraph(bee_tidy, layout = "graphopt") + 
    geom_node_point() +
    geom_edge_link(aes(width = weight), alpha = 0.3) + 
    scale_edge_width(range = c(0.2, 2)) +
    geom_node_text(aes(label = name), repel = TRUE) +
    labs(edge_width = "weight") +
    theme_graph() #Graph the network!
  degreecentOH <- as.data.frame(degree_w(edges))
  degreecentOH$name <- nodes$name
}


