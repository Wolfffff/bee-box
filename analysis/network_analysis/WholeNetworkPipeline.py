import sys
import os
import random
import pandas as pd
import itertools as it
import networkx as nx
import networkx.algorithms.community as nxcom
import matplotlib.pyplot as plt
import os
import argparse
import numpy
import scipy
import seaborn as sns
from scipy import stats


def set_node_community(G, communities):
    """Add community to node attributes"""
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]["community"] = c + 1


def set_edge_community(G):
    """Find internal edges and add their community to their attributes"""
    for (
        v,
        w,
    ) in G.edges:
        if G.nodes[v]["community"] == G.nodes[w]["community"]:
            # Internal edge, mark with community
            G.edges[v, w]["community"] = G.nodes[v]["community"]
        else:
            # External edge, mark as 0
            G.edges[v, w]["community"] = 0


def get_color(i, r_off=1, g_off=1, b_off=1):
    """Assign a color to a vertex."""
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)


def parser():
    """
    Argument parser
    Raises
    ------
    IOError
            If the input, or other specified files do not exist
    """

    def confirmDir():
        """Custom action to confirm file exists"""

        class customAction(argparse.Action):
            def __call__(self, parser, args, value, option_string=None):
                if not os.path.isdir(value):
                    raise IOError("%s not found" % value)
                setattr(args, self.dest, value)

        return customAction

    def confirmFile():
        """Custom action to confirm file exists"""

        class customAction(argparse.Action):
            def __call__(self, parser, args, value, option_string=None):
                if not os.path.isfile(value):
                    raise IOError("%s not found" % value)
                setattr(args, self.dest, value)

        return customAction

    my_parser = argparse.ArgumentParser()
    # Input arguments
    my_parser.add_argument(
        "--dir", help="input directory with networks", type=str, action=confirmDir()
    )
    my_parser.add_argument(
        "--subgroup", help="subgroup of interest", type=str, action=confirmFile()
    )
    my_parser.add_argument(
        "--filter", help="nodes to not filter out", type=str, action=confirmFile()
    )
    centtypes = ["betweenness", "degree", "eigen", "closeness"]
    my_parser.add_argument(
        "--centrality", help="", type=str, choices=centtypes, default="degree"
    )
    delimtypes = [",", "\t"]
    my_parser.add_argument(
        "--delim", help="", type=str, choices=delimtypes, default="\t"
    )
    my_parser.add_argument("--thresh", help="", type=float, default=0.05)
    analtypes = ["all", "centrality", "assortativity", "community"]
    my_parser.add_argument(
        "--analysis",
        help="which analyses you want",
        type=str,
        choices=analtypes,
        default="all",
    )

    return my_parser.parse_args()


def centrality(G2):
    print("Running Centrality Module")
    # Check the type of centrality, and calculate for each node
    if myargs.centrality == "degree":
        cent = nx.degree_centrality(G2)
        cent_size = numpy.fromiter(cent.values(), float)
        print(cent)
    if myargs.centrality == "eigen":
        cent = nx.eigenvector_centrality(G2)
        cent_size = numpy.fromiter(cent.values(), float)
        print(cent)
    if myargs.centrality == "betweenness":
        cent = nx.betweenness_centrality(G2)
        cent_size = numpy.fromiter(cent.values(), float)
        print(cent)
    if myargs.centrality == "closeness":
        cent = nx.closeness_centrality(G2)
        cent_size = numpy.fromiter(cent.values(), float)
        print(cent)

    # This gives a degree frequency index (useful to compare to Power Law)
    degree_G2 = nx.degree(G2)
    degree_df = pd.DataFrame(degree_G2, columns=["Node", "Degree"])
    degree_list = degree_df["Degree"].to_numpy()
    degree_freq_G2 = nx.degree_histogram(G2)
    degree_freq_df = pd.DataFrame(degree_freq_G2, columns=["Frequency"])
    degree_freq_df["Degree"] = degree_freq_df.index
    degree_freq_df = degree_freq_df[["Degree", "Frequency"]]
    degree_df.to_csv(f"{edge_file}_Freq.txt", sep="\t")

    # This allows us to compare centrality between two different
    Cents1 = []
    Cents0 = []
    for v in G2.nodes:
        if v in group:
            G2.nodes[v]["subgroup"] = 1
            G2.nodes[v]["centrality"] = cent[v]
            Cents1.append(cent[v])
        else:
            G2.nodes[v]["subgroup"] = 0
            G2.nodes[v]["centrality"] = cent[v]
            Cents0.append(cent[v])
    node_color = [get_color(G2.nodes[v]["subgroup"]) for v in G2.nodes]
    # print(G2.nodes)
    # print(Cents1)
    # print(Cents0)

    # Output1: Graph, Highlight High Centrality & Groups.
    plt.figure()
    nx.draw(
        G2,
        pos=None,
        with_labels=True,
        node_color=node_color,
        node_size=cent_size * 2000,
        width=1,
    )  # ,ax=fig.subplot(111))
    plt.savefig(f"{myargs.centrality}_{myargs.thresh}_{edge_file}_Network.png")
    # plt.show()

    # Output2: Degree Histogram
    fig = plt.figure("Degree of Graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    # axgrid = fig.add_gridspec(5, 4)

    # ax2 = fig.add_subplot(axgrid[:, :])
    # ax2.bar(*numpy.unique(degree_sequence, return_counts=True))
    # ax2.set_title("Degree histogram")
    # ax2.set_xlabel("Degree")
    # ax2.set_ylabel("# of Nodes")
    histoimage = plt.hist(cent.values(), range=[0, 0.15], color="skyblue")
    fig.tight_layout()
    fig.savefig(f"{myargs.centrality}_{myargs.thresh}_{edge_file}_Histo.png")
    # plt.show()


def assortativity(G2):
    for v in G2.nodes:
        if v in group:
            G2.nodes[v]["subgroup"] = 1
        else:
            G2.nodes[v]["subgroup"] = 0
    assortativity_G2 = nx.degree_assortativity_coefficient(G2)
    print("Degree Assortativity:")
    print(assortativity_G2)
    attrassort_G2 = nx.attribute_assortativity_coefficient(G2, "subgroup")
    print("Subgroup Assortativity:")
    print(attrassort_G2)


def community(G2, heat_map):
    communities = sorted(nxcom.greedy_modularity_communities(G2), key=len, reverse=True)
    print(communities)
    print(G2.nodes)
    for c in communities:
        for v, w in it.combinations(c, 2):
            heat_map[v][w] += 1
            heat_map[w][v] += 1
    return heat_map


myargs = parser()
# print(os.listdir(myargs.dir))
print("Loading in...")
with open(myargs.filter) as f:
    lines = f.readlines()
a = [eval(x) for x in lines]
heat_map = pd.DataFrame(0, index=list(a), columns=list(a))
for pos, edge_file in enumerate(os.listdir(myargs.dir)):
    # Read in a single file in the folder (We'll get back to the others!)
    if "csv" not in edge_file:
        continue
    edge_path = os.path.join(myargs.dir, edge_file)
    # print(edge_path)
    Delimeter = myargs.delim
    Weights = pd.read_table(edge_path, delimiter=Delimeter)
    # print(Weights)

    # Filter by Threshold & by Tag List
    WeightsThresh = Weights[Weights[Weights.columns[2]] > float(myargs.thresh)]
    WeightsThresh.columns = ["source", "target", "Weight"]
    # print(WeightsThresh)
    if myargs.filter:
        # print(a)
        WeightsBool = WeightsThresh.source.isin(a)
        # print(WeightsBool)
        WeightsThresh = WeightsThresh[WeightsBool]
        # print(WeightsThresh)
        WeightsBool = WeightsThresh.target.isin(a)
        WeightsFiltered = WeightsThresh[WeightsBool]
    else:
        WeightsFiltered = WeightsThresh
    # print(WeightsFiltered)

    G2 = nx.from_pandas_edgelist(WeightsFiltered)
    with open(myargs.subgroup) as f:
        lines = f.readlines()
    group = [eval(x) for x in lines]
    # print(group)
    # print(G2.nodes)

    ##Centrality Measures (4 Main Kinds)
    if myargs.analysis == "all" or myargs.analysis == "centrality":
        centrality(G2)

    ##Associativity
    if myargs.analysis == "all" or myargs.analysis == "assortativity":
        assortativity(G2)

    # Between High/Low Centrality

    # Compare Between Groups

    # Output Assortativity: Regressed Scatterplots; Look in Seaborn

    ##Community Structure
    if myargs.analysis == "all" or myargs.analysis == "community":
        print(heat_map)
        heat_map = community(G2, heat_map)


if myargs.analysis == "all" or myargs.analysis == "community":
    HeatMapFig = sns.clustermap(heat_map.values)
    HeatMapFig.figure.savefig("HeatMap.png")
