setwd("d:\\matching_data")
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
# Read CSV file as dataframe
OH <- read.csv('20211018_24k_pipeline_run_aruco_data_with_track_numbers.csv', header=TRUE)
# Get rid of rows that have centroid NaNs
OH <- OH[complete.cases(OH[ , 3:4]),]
# Get rid of tag -1 (?)
OH <- OH %>% filter(Tag!=-1) 
# Forget about columns we won't use from here on
# Frame, Tag, cX, xY
OHNEdited<- OH[,1:4] 

# Write trimmed down coordinates as .tsv file
write.table(OHNEdited, "GreedyTags.tsv", row.names = F, sep=",") #ExportToRunAndrewsJob

OHN <- read.table('350PixelCutOff.tsv', header=FALSE) #Reads Andrew's Export
colnames(OHN) <- c('to', 'from', 'weight')
OHN <- OHN %>% filter(weight>0.05) #Only interacting 2% of the time or more
edges_g <- as_tbl_graph(OHN)
nodes = edges_g %>% 
  activate(nodes) %>% 
  as_tibble() %>% 
  print(n=10)
nodes$Xav <- 0
nodes$Yav <- 0
nodes$DistanceFromCenter <- 0
meanX<-mean(OHNEdited$cX) 
meanY<-mean(OHNEdited$cY) 
for(i in 1:nrow(nodes)){
  t <- nodes$name[i] 
  nodes$Xav[i] <- mean(subset(OHNEdited, Tag==t)$cX)
  nodes$Yav[i] <- mean(subset(OHNEdited, Tag==t)$cY)
  nodes$DistanceFromCenter[i] <- sqrt((nodes$Xav[i] - meanX)^2 + (nodes$Yav[i] - meanY)^2) #Base Measures of Centrality
}
edges = edges_g %>% 
  activate(edges) %>% 
  as_tibble() %>% print(n=10)
bee_tidy <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE) #Turn into graph format
ggraph(bee_tidy, layout = "graphopt") + 
  geom_node_point() +
  geom_edge_link(aes(width = weight), alpha = 0.3) + 
  scale_edge_width(range = c(0.2, 2)) +
  geom_node_text(aes(label = name, color=DistanceFromCenter), repel = TRUE) +
  labs(edge_width = "weight") +
  theme_graph() #These steps all just turn it to a graph
CentralitiesUncut <- as.data.frame(nodes[,c(1)])
degreecentUncut <- centr_degree(bee_tidy)
CentralitiesUncut$CentUncut <- degreecentUncut$res
CentralitiesList <- list(CentralitiesUncut)
#"Cut Deck" Theoretical Data
setwd("~/Downloads/24k_pipeline_run_greedy/24k_pipeline_run_greedy/")
OHS <- read.csv('20211018_24k_pipeline_run_aruco_data_with_track_numbers.csv', header=TRUE)
OHS <- OHS[complete.cases(OHS[ , 3:4]),] #Remove missing data
OHS <- OHS %>% filter(Tag!=-1) #Only interacting 10% of the time or more
OHSEdited<- OHS[,1:4]
Mn <- min(OHSEdited$Frame)
Mx <- max(OHSEdited$Frame)
MxTag <- max(OHSEdited$Tag)
Rg <- Mx - Mn 
setwd("~/Downloads")
for(j in 1:100){
  Offsets <- sample(Rg, MxTag+1, replace=TRUE) #Generate a random offset for each individual
  OHSEdited$i <- OHSEdited$Tag + 1
  OHSEdited$OffSetFrame <- OHSEdited$Frame + Offsets[OHSEdited$i] #Move forward by the offset
  OHSEdited[OHSEdited$OffSetFrame>Mx,]$OffSetFrame <- OHSEdited[OHSEdited$OffSetFrame>Mx,]$OffSetFrame - Rg #Shift back any that goes outside the range
  OHSExport = subset(OHSEdited, select=c(OffSetFrame, Tag, cX, cY))
  colnames(OHSExport) <- c("Frame", "Tag", "cX", "cY")
  OHSExport <- OHSExport[order(OHSExport$Frame),]
  write.table(OHSExport, sep=",", file = paste0("24KN/CutDeckData",".",deparse(j), ".tsv"),row.names = FALSE)
  print(i)
}

#Andrew Analysis

setwd("~/Downloads/24KNO")
temp = list.files(pattern="*DeckData*")
myfiles = lapply(temp, read.table,header=FALSE) #Loads all theoretical networks
Data <- myfiles %>% reduce(left_join, by = c("V1","V2"))
colnames(Data) <- c("to","from",c(1:100))
Data$mean <- rowMeans(Data[,3:101], na.rm=T)

for(i in 1:100){
  OHCut <- Data %>% select(1,2,i+2)
  colnames(OHCut) <- c('to', 'from', 'weight')
  OHCut <- OHCut[complete.cases(OHCut),]
  OHCut <- OHCut %>% filter(weight>0.05)
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
  CentralitiesCut <- as.data.frame(nodes[,c(1)])
  degreecentCut <- centr_degree(bee_tidy)
  CentralitiesCut$CentCut <- degreecentCut$res
  CentralitiesList[[i+1]] <- CentralitiesCut
}
Centralities <- CentralitiesList %>% reduce(left_join, by = c("name"))
colnames(Centralities) <- c("name","Uncut",c(1:100))
sort(Centralities[5,c(2:101)])
Influence <- Centralities %>% mutate(Influence = Uncut/rowMeans(.[, 3:51]))
Freqs <- as.data.frame(table(OH$Tag)) %>% filter(Freq>0.1*max(OH$Frame))
Influence <- Influence %>% filter(name %in% Freqs$Var1) 
ggplot(Influence, aes(x=Influence)) + 
  geom_histogram(binwidth=0.1) + 
  geom_vline(xintercept=mean(Influence$Influence), linetype="dashed", 
             color = "red", size=2)