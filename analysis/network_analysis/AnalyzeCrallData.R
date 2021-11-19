setwd("d:\\matching_data")
# Load Directories --------------------------------------------------------
library(igraph)
library(tidygraph)
library(ggraph)
# Impute NaNs -------------------------------------------------------------
NearMean <- function(M5, M4, M3, M2, M1, P1, P2, P3, P4, P5){
  UW=NaN
  LB=NaN
  LW=NaN
  UB=NaN
  if (!is.na(M1)){
    LB = M1
    LW = 1
  }
  else{
    if (!is.na(M2)){
      LB = M2
      LW = 2
    }
    else{
      if (!is.na(M3)){
        LB = M3
        LW = 3
      }
      else{
        if (!is.na(M4)){
          LB = M4
          LW = 4
        }
        else{
          if (!is.na(M5)){
            LB = M5
            LW = 5
          }
        }
      }
    }
  }
  if (!is.na(P1)){
    UB = P1
    UW = 1
  }
  else{
    if (!is.na(P2)){
      UB = P2
      UW = 2
    }
    else{
      if (!is.na(P3)){
        UB = P3
        UW = 3
      }
      else{
        if (!is.na(P4)){
          UB = P4
          UW = 4
        }
        else{
          if (!is.na(P5)){
            UB = P5
            UW = 5
          }
        }
      }
    }
  }
  X = ((UW*LB)+(LW*UB))/(UW + LW)
  return(X)
} #This Function Takes the Average of the nearest lower value and nearest upper value

TD <- read.delim('20211018_24k_pipeline_run_aruco_data_with_track_numbers.csv', sep = ",", header=F) #Read the two files
TD <- TD[,c(1:15, 17:24)]
TDY <- read.delim('TDY.txt', sep = ",", header=F)
TDY <- TDY[,c(1:15, 17:24)]
#Next 6 Lines create "blank" files to play with
TDImp <- TD
TDYImp <- TDY
TDVel <- TD
TDVel[1,] <- NaN
TDYVel <- TDY
TDYVel[1,] <- NaN
#For X Axis, imputes unknown values
for(i in 1:nrow(TD)){
  for(j in 1:ncol(TD)){
    if(is.na(TD[i,j])){
      if(i>5 & i<nrow(TD)-5){
        X <- NearMean(TD[i-5,j], TD[i-4,j], TD[i-3,j], TD[i-2,j], TD[i-1,j], TD[i+1,j], TD[i+2,j], TD[i+3,j], TD[i+4,j], TD[i+5,j])
        TDImp[i,j] = X
      }
    }
  }
}
#For Y Axis, imputes unknown values
for(i in 1:nrow(TDY)){
  for(j in 1:ncol(TDY)){
    if(is.na(TDY[i,j])){
      if(i>5 & i<nrow(TDY)-5){
        X <- NearMean(TDY[i-5,j], TDY[i-4,j], TDY[i-3,j], TDY[i-2,j], TDY[i-1,j], TDY[i+1,j], TDY[i+2,j], TDY[i+3,j], TDY[i+4,j], TDY[i+5,j])
        TDYImp[i,j] = X
      }
    }
  }
}
#For X Axis, predict movement/frame
for(i in 2:nrow(TDImp)){
  for(j in 1:ncol(TDImp)){
    X <- TDImp[i,j] - TDImp[i-1,j]
    TDVel[i,j] = X
  }
}
#For Y Axis, predicts movement/frame
for(i in 2:nrow(TDYImp)){
  for(j in 1:ncol(TDYImp)){
    X <- TDYImp[i,j] - TDYImp[i-1,j]
    TDYVel[i,j] = X
  }
}

# Empicial Network Analysis -----------------------------------------------



CutOff <- 300 #Distance to be considered an interaction
nodes <- seq(1,ncol(TDY)) #List of Bees
edges <- as.data.frame(t(combn(nodes, 2))) #Data Frame of number of interactions in each pair
Weights <- rep(0, nrow(edges)) 
edges <- cbind(edges, Weights) #Add a Weights Column, which is the amount of time two bees are interacting
colnames(edges) <- c("from", "to", "weight")
BeeNames <- c("Bee1", "Bee2", "Bee3", "Bee4", "Bee5", "Bee6", "Bee7", "Bee8", "Bee9", "Bee10", "Bee11", "Bee12", "Bee13", "Bee14", "Bee15", "Bee16", "Bee17", "Bee18", "Bee19", "Bee20", "Bee21", "Bee22", "Bee23")
nodes <- as.data.frame(cbind(nodes, BeeNames))
colnames(nodes) <- c("id", "label")
nodes$id <-  as.integer(nodes$id)
nodes$label <-  as.character(nodes$label) #All of this is just getting it all in the right format for tidygraph
for(i in 2:nrow(TDYImp)){
  for(x in 1:nrow(edges)){
    A <- edges[x,1]
    B <- edges[x,2]
    if(is.nan(TDYImp[i,A]) == F & is.nan(TDYImp[i,B]) == F){
      if(sqrt((TDImp[i,A] - TDImp[i,B])^2 + (TDYImp[i,A] - TDYImp[i,B])^2) < CutOff){
        edges[x,3] <- edges[x,3] + 1
      }
    }
  }
} #Identify every single pair within 300 pixels of each other. 
edges_filtered <- edges[ which(edges$weight > 75),] #Filter inteactions of less than 75 frames (otherwise the graph is ugly)
bee_tidy <- tbl_graph(nodes = nodes, edges = edges_filtered, directed = FALSE) #Turn into graph format
ggraph(bee_tidy, layout = "graphopt") + 
  geom_node_point() +
  geom_edge_link(aes(width = weight), alpha = 0.8) + 
  scale_edge_width(range = c(0.2, 2)) +
  geom_node_text(aes(label = label), repel = TRUE) +
  labs(edge_width = "weight") +
  theme_graph() #Graph the network!

# Self-Absorbed Shuffled Network --------------------------------------------------------

#Simulate a network
TDSim <- TDImp #Create a Base for the simulated x-axis data
TDYSim <- TDYImp #Create a Base for the simulated y-axis data
#XAxis
for(i in 1:ncol(TDSim)){
  Locs <- na.omit(unique(TDImp[,i]))
  Vels <- na.omit(unique(TDVel[,i]))
  LocMin <- min(Locs)
  LocMax <- max(Locs)
  ToNan=0
  ToDat=0
  for(x in 2:nrow(TDSim)){
    if((is.nan(TDImp[x,i]) & !is.nan(TDImp[x-1,i]))){
      ToNan = ToNan+1
    }
    if((!is.nan(TDImp[x,i]) & is.nan(TDImp[x-1,i]))){
      ToDat = ToDat+1
    }
  }
  ToNan <- ToNan / sum(!is.na(TDImp[,i]))
  ToDat <- ToDat / sum(is.na(TDImp[,i]))
  TDSim[1,i] = sample(Locs,1)
  State <- sample(c(0,1), 1)
  for(j in 2:nrow(TDSim)){
    TDSim[j,i] = TDSim[j-1,i] + sample(Vels,1)
    if(TDSim[j,i] >= LocMax){
      TDSim[j,i] = LocMax
    }
    if(TDSim[j,i] <= LocMin){
      TDSim[j,i] = LocMin
    }
  }
}
#YAxis
for(i in 1:ncol(TDYSim)){
  Locs <- na.omit(unique(TDYImp[,i]))
  Vels <- na.omit(unique(TDYVel[,i]))
  LocMin <- min(Locs)
  LocMax <- max(Locs)
  ToNan=0
  ToDat=0
  for(x in 2:nrow(TDYSim)){
    if((is.nan(TDYImp[x,i]) & !is.nan(TDYImp[x-1,i]))){
      ToNan = ToNan+1
    }
    if((!is.nan(TDYImp[x,i]) & is.nan(TDYImp[x-1,i]))){
      ToDat = ToDat+1
    }
  }
  ToNan <- ToNan / sum(!is.na(TDYImp[,i]))
  ToDat <- ToDat / sum(is.na(TDYImp[,i]))
  TDYSim[1,i] = sample(Locs,1)
  State <- sample(c(0,1), 1)
  for(j in 2:nrow(TDYSim)){
    TDYSim[j,i] = TDYSim[j-1,i] + sample(Vels,1)
    if(TDYSim[j,i] >= LocMax){
      TDYSim[j,i] = LocMax
    }
    if(TDYSim[j,i] <= LocMin){
      TDYSim[j,i] = LocMin
    }
    if(State%%2 == 0){
      TDYSim[j-1,i] = NaN
      TDSim[j-1,i] = NaN
    }
    r <- runif(1)
    if(State%%2 == 0){
      if(r < ToDat){
        State = State + 1
      }
    }
    else{
      if(r < ToNan){
        State = State + 1
      }
    }
  }
}

#SameAnalysisOnSimulatedData
CutOff <- 300 #Distance to be considered an interaction
nodesSim <- seq(1,ncol(TDYSim)) #List of Bees
edgesSim <- as.data.frame(t(combn(nodesSim, 2))) #Data Frame of number of interactions in each pair
WeightsSim <- rep(0, nrow(edgesSim)) 
edgesSim <- cbind(edgesSim, WeightsSim) #Add a Weights Column, which is the amount of time two bees are interacting
colnames(edgesSim) <- c("from", "to", "weight")
BeeNamesSim <- c("Bee1", "Bee2", "Bee3", "Bee4", "Bee5", "Bee6", "Bee7", "Bee8", "Bee9", "Bee10", "Bee11", "Bee12", "Bee13", "Bee14", "Bee15", "Bee16", "Bee17", "Bee18", "Bee19", "Bee20", "Bee21", "Bee22", "Bee23")
nodesSim <- as.data.frame(cbind(nodesSim, BeeNamesSim))
colnames(nodesSim) <- c("id", "label")
nodesSim$id <-  as.integer(nodesSim$id)
nodesSim$label <-  as.character(nodesSim$label) #All of this is just getting it all in the right format for tidygraph
for(i in 2:nrow(TDYSim)){
  for(x in 1:nrow(edgesSim)){
    A <- edgesSim[x,1]
    B <- edgesSim[x,2]
    if(is.nan(TDYSim[i,A]) == F & is.nan(TDSim[i,B]) == F){
      if(sqrt((TDSim[i,A] - TDSim[i,B])^2 + (TDYSim[i,A] - TDYSim[i,B])^2) < CutOff){
        edgesSim[x,3] <- edgesSim[x,3] + 1
      }
    }
  }
} #Identify every single pair within 300 pixels of each other. 
edges_filteredSim <- edgesSim[ which(edgesSim$weight > 75),] #Filter inteactions of less than 75 frames (otherwise the graph is ugly)
bee_tidy <- tbl_graph(nodes = nodesSim, edges = edges_filteredSim, directed = FALSE) #Turn into graph format
ggraph(bee_tidy, layout = "graphopt") + 
  geom_node_point() +
  geom_edge_link(aes(width = weight), alpha = 0.8) + 
  scale_edge_width(range = c(0.2, 2)) +
  geom_node_text(aes(label = label), repel = TRUE) +
  labs(edge_width = "weight") +
  theme_graph() #Graph the network!


# Brownian Network --------------------------------------------------------


#Simulate a Brownian network -- X Axis
TDBro <- TDImp #Create a Base for the simulated x-axis data
TDYBro <- TDYImp #Create a Base for the simulated y-axis data

#XAxis

Locs <- as.data.frame(na.omit(unique(TDImp[,1])))
colnames(Locs) <- 'Locations'
for(i in 2:ncol(TDYImp)){
  LocsAdd <- as.data.frame(na.omit(unique(TDImp[,i])))
  colnames(LocsAdd) <- 'Locations'
  Locs <- rbind(Locs, LocsAdd)
}
Locs <- unique(Locs)
Vels <- as.data.frame(na.omit(unique(TDVel[,1])))
colnames(Vels) <- 'Velocities'
for(i in 2:ncol(TDImp)){
  VelsAdd <- as.data.frame(na.omit(unique(TDVel[,i])))
  colnames(VelsAdd) <- 'Velocities'
  Vels <- rbind(Vels, VelsAdd)
}
Vels <- unique(Vels)
LocMin <- min(Locs$Locations)
LocMax <- max(Locs$Locations)
ToNan=0
ToDat=0
for(i in 1:ncol(TDYImp)){
  for(x in 2:nrow(TDYSim)){
    if((is.nan(TDYImp[x,1]) & !is.nan(TDYImp[x-1,1]))){
      ToNan = ToNan+1
    }
    if((!is.nan(TDYImp[x,1]) & is.nan(TDYImp[x-1,1]))){
      ToDat = ToDat+1
    }
  }
}
ToNan <- ToNan / sum(!is.na(TDYImp))
ToDat <- ToDat / sum(is.na(TDYImp))
State <- sample(c(0,1), 1)
for(i in 1:ncol(TDBro)){
  TDBro[1,i] = sample(Locs$Locations,1)
  for(j in 2:nrow(TDBro)){
    TDBro[j,i] = TDBro[j-1,i] + sample(Vels$Velocities,1)
    if(TDBro[j,i] >= LocMax){
      TDBro[j,i] = LocMax
    }
    if(TDBro[j,i] <= LocMin){
      TDBro[j,i] = LocMin
    }
  }
}

#Y-Axis Now!
Locs <- as.data.frame(na.omit(unique(TDYImp[,1])))
colnames(Locs) <- 'Locations'
for(i in 2:ncol(TDYImp)){
  LocsAdd <- as.data.frame(na.omit(unique(TDYImp[,i])))
  colnames(LocsAdd) <- 'Locations'
  Locs <- rbind(Locs, LocsAdd)
}
Locs <- unique(Locs)
Vels <- as.data.frame(na.omit(unique(TDYVel[,1])))
colnames(Vels) <- 'Velocities'
for(i in 2:ncol(TDYImp)){
  VelsAdd <- as.data.frame(na.omit(unique(TDYVel[,i])))
  colnames(VelsAdd) <- 'Velocities'
  Vels <- rbind(Vels, VelsAdd)
}
Vels <- unique(Vels)
LocMin <- min(Locs$Locations)
LocMax <- max(Locs$Locations)
TransFreq=0
for(x in 2:nrow(TDYBro)){
  if((is.nan(TDYImp[x,1]) & !is.nan(TDYImp[x-1,1])) | (!is.nan(TDYImp[x,1]) & is.nan(TDYImp[x-1,1]))){
    TransFreq = TransFreq+1
  }
}
TransFreq <- TransFreq / (nrow(TDYSim)-1)
State <- sample(c(0,1), 1)
for(i in 1:ncol(TDYBro)){
  TDYBro[1,i] = sample(Locs$Locations,1)
  for(j in 2:nrow(TDYBro)){
    TDYBro[j,i] = TDYBro[j-1,i] + sample(Vels$Velocities,1)
    if(TDYBro[j,i] >= LocMax){
      TDYBro[j,i] = LocMax
    }
    if(TDYBro[j,i] <= LocMin){
      TDYBro[j,i] = LocMin
    }
    if(State%%2 == 0){
      TDYBro[j-1,i] = NaN
      TDBro[j-1,i] = NaN
    }
    r <- runif(1)
    if(State%%2 == 0){
      if(r < ToDat){
        State = State + 1
      }
    }
    else{
      if(r < ToNan){
        State = State + 1
      }
    }
  }
}

#SameAnalysisOnBrownianData
CutOff <- 300 #Distance to be considered an interaction
nodesBro <- seq(1,ncol(TDYBro)) #List of Bees
edgesBro <- as.data.frame(t(combn(nodesBro, 2))) #Data Frame of number of interactions in each pair
WeightsBro <- rep(0, nrow(edgesBro)) 
edgesBro <- cbind(edgesBro, WeightsBro) #Add a Weights Column, which is the amount of time two bees are interacting
colnames(edgesBro) <- c("from", "to", "weight")
BeeNamesBro <- c("Bee1", "Bee2", "Bee3", "Bee4", "Bee5", "Bee6", "Bee7", "Bee8", "Bee9", "Bee10", "Bee11", "Bee12", "Bee13", "Bee14", "Bee15", "Bee16", "Bee17", "Bee18", "Bee19", "Bee20", "Bee21", "Bee22", "Bee23")
nodesBro <- as.data.frame(cbind(nodesBro, BeeNamesBro))
colnames(nodesBro) <- c("id", "label")
nodesBro$id <-  as.integer(nodesBro$id)
nodesBro$label <-  as.character(nodesBro$label) #All of this is just getting it all in the right format for tidygraph
for(i in 2:nrow(TDYBro)){
  for(x in 1:nrow(edgesBro)){
    A <- edgesBro[x,1]
    B <- edgesBro[x,2]
    if(is.nan(TDYBro[i,A]) == F & is.nan(TDBro[i,B]) == F){
      if(sqrt((TDBro[i,A] - TDBro[i,B])^2 + (TDYBro[i,A] - TDYBro[i,B])^2) < CutOff){
        edgesBro[x,3] <- edgesBro[x,3] + 1
      }
    }
  }
} #Identify every single pair within 300 pixels of each other. 
edges_filteredBro <- edgesBro[ which(edgesBro$weight > 75),] #Filter inteactions of less than 75 frames (otherwise the graph is ugly)
bee_tidy <- tbl_graph(nodes = nodesBro, edges = edges_filteredBro, directed = FALSE) #Turn into graph format
ggraph(bee_tidy, layout = "graphopt") + 
  geom_node_point() +
  geom_edge_link(aes(width = weight), alpha = 0.8) + 
  scale_edge_width(range = c(0.2, 2)) +
  geom_node_text(aes(label = label), repel = TRUE) +
  labs(edge_width = "weight") +
  theme_graph() #Graph the network!


# Translated Network ------------------------------------------------------


#Simulate a Translated network -- X Axis
TDTra <- TDImp #Create a Base for the simulated x-axis data
TDYTra <- TDYImp #Create a Base for the simulated y-axis data

#XAndYAxis

for(i in 2:ncol(TDTra)){
  R <- sample(1:nrow(TDImp), 1)
  for(j in 1:nrow(TDTra)){
    x <- R%%1946 + 1
    TDTra[j,i] <- TDImp[x,i]
    TDYTra[j,i] <- TDYImp[x,i]
    R = R + 1
  }
}

#SameAnalysisOnTranslatedData
CutOff <- 300 #Distance to be considered an interaction
nodesTra <- seq(1,ncol(TDYTra)) #List of Bees
edgesTra <- as.data.frame(t(combn(nodesTra, 2))) #Data Frame of number of interactions in each pair
WeightsTra <- rep(0, nrow(edgesTra)) 
edgesTra <- cbind(edgesTra, WeightsTra) #Add a Weights Column, which is the amount of time two bees are interacting
colnames(edgesTra) <- c("from", "to", "weight")
BeeNamesTra <- c("Bee1", "Bee2", "Bee3", "Bee4", "Bee5", "Bee6", "Bee7", "Bee8", "Bee9", "Bee10", "Bee11", "Bee12", "Bee13", "Bee14", "Bee15", "Bee16", "Bee17", "Bee18", "Bee19", "Bee20", "Bee21", "Bee22", "Bee23")
nodesTra <- as.data.frame(cbind(nodesTra, BeeNamesTra))
colnames(nodesTra) <- c("id", "label")
nodesTra$id <-  as.integer(nodesTra$id)
nodesTra$label <-  as.character(nodesTra$label) #All of this is just getting it all in the right format for tidygraph
for(i in 2:nrow(TDYTra)){
  for(x in 1:nrow(edgesTra)){
    A <- edgesTra[x,1]
    B <- edgesTra[x,2]
    if(is.nan(TDYTra[i,A]) == F & is.nan(TDTra[i,B]) == F){
      if(sqrt((TDTra[i,A] - TDTra[i,B])^2 + (TDYTra[i,A] - TDYTra[i,B])^2) < CutOff){
        edgesTra[x,3] <- edgesTra[x,3] + 1
      }
    }
  }
} #Identify every single pair within 300 pixels of each other. 
edges_filteredTra <- edgesTra[ which(edgesTra$weight > 75),] #Filter inteactions of less than 75 frames (otherwise the graph is ugly)
bee_tidy <- tbl_graph(nodes = nodesTra, edges = edges_filteredTra, directed = FALSE) #Turn into graph format
ggraph(bee_tidy, layout = "graphopt") + 
  geom_node_point() +
  geom_edge_link(aes(width = weight), alpha = 0.8) + 
  scale_edge_width(range = c(0.2, 2)) +
  geom_node_text(aes(label = label), repel = TRUE) +
  labs(edge_width = "weight") +
  theme_graph() #Graph the network!


