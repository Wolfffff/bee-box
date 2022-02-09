import sys
import os
import random
import pandas as pd
import itertools as it
import networkx as nx
import networkx.algorithms.community as nxcom
import matplotlib.pyplot as plt
import os
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


for pos, edge_file in enumerate(os.listdir(sys.argv[1])):
    edge_path = os.path.join(sys.argv[1], edge_file)
    Weights = pd.read_table(edge_path)
    print("Loading...")
    # print(Weights)
    a = (
        0,
        1,
        2,
        5,
        6,
        8,
        10,
        12,
        13,
        14,
        17,
        18,
        19,
        20,
        22,
        24,
        27,
        30,
        34,
        35,
        38,
        40,
        42,
        44,
        47,
        48,
        49,
        51,
        52,
        53,
        54,
        55,
        56,
        58,
        60,
        61,
        62,
        65,
        70,
        71,
        73,
        74,
        75,
    )
    WeightsThresh = Weights[Weights[Weights.columns[2]] > float(sys.argv[2])]
    WeightsThresh.columns = ["source", "target", "Weight"]
    WeightsBool = WeightsThresh.source.isin(a)
    WeightsThresh = WeightsThresh[WeightsBool]
    WeightsBool = WeightsThresh.target.isin(a)
    WeightsFiltered = WeightsThresh[WeightsBool]
    G2 = nx.from_pandas_edgelist(WeightsFiltered)
    density = nx.density(G2)
    # print(density)
    degree_G2 = nx.degree(G2)
    degree_df = pd.DataFrame(degree_G2, columns=["Node", "Degree"])
    # print(degree_df)
    degree_list = degree_df["Degree"].to_numpy()
    # print(degree_list)
    degree_freq_G2 = nx.degree_histogram(G2)
    degree_freq_df = pd.DataFrame(degree_freq_G2, columns=["Frequency"])
    degree_freq_df["Degree"] = degree_freq_df.index
    degree_freq_df = degree_freq_df[["Degree", "Frequency"]]
    degree_df.to_csv("Frequency_Conc_Filt.tsv", sep="\t")
    # nx.draw(G2, with_labels=True)
    # plt.show()
    assortativity_G2 = nx.degree_assortativity_coefficient(G2)
    # print(assortativity_G2)
    node_mapping = dict(
        zip(G2.nodes(), sorted(G2.nodes(), key=lambda k: random.random()))
    )
    G2_relabeled = nx.relabel_nodes(
        G2, node_mapping
    )  # This Randomizes the nodes for bootstrapping
    communities = sorted(
        nxcom.greedy_modularity_communities(G2_relabeled), key=len, reverse=True
    )
    # print(communities)
    combs = []
    if pos == 0:
        heat_map = pd.DataFrame(0, index=G2.nodes, columns=G2_relabeled.nodes)
    for c in communities:
        for v, w in it.combinations(c, 2):
            heat_map[v][w] += 1
            heat_map[w][v] += 1
    # print(heat_map[38][38])
    # print(heat_map.values)
    set_node_community(G2, communities)
    set_edge_community(G2)
    node_color = [get_color(G2.nodes[v]["community"]) for v in G2.nodes]
    external = [(v, w) for v, w in G2.edges if G2.edges[v, w]["community"] == 0]
    internal = [(v, w) for v, w in G2.edges if G2.edges[v, w]["community"] > 0]
    internal_color = ["black" for e in internal]
    bee_pos = nx.spring_layout(G2)
    plt.rcParams.update({"figure.figsize": (15, 10)})
    # Draw external edges
    nx.draw_networkx(
        G2, pos=bee_pos, node_size=0, edgelist=external, edge_color="silver"
    )
    # Draw nodes and internal edges
    nx.draw_networkx(
        G2,
        pos=bee_pos,
        node_color=node_color,
        edgelist=internal,
        edge_color=internal_color,
    )
    plt.show()
    S = stats.kstest(degree_list, "powerlaw", args=(8.25, 21))
    print(S)
    ax = sns.clustermap(heat_map.values)
# fig = ax.get_figure()
ax.savefig("robustness_heatmap_bootstrap.pdf")
# print(S)
# print(assortativity_G2)
# print(Weights)
# print(sys.argv)
# a = (0, 1, 2, 5, 6, 8, 10, 12, 13, 14, 17, 18, 19, 20, 22, 24, 27, 30, 34, 35, 38, 40, 42, 44, 47, 48, 49, 51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 65, 70, 71, 73, 74, 75)
# print(a)
# WeightsThresh = Weights[Weights[Weights.columns[2]] > float(sys.argv[2])]
# print(WeightsThresh)
# WeightsThresh.columns =['source', 'target', 'Weight']
# WeightsBool = WeightsThresh.source.isin(a)
# WeightsThresh = WeightsThresh[WeightsBool]
# WeightsBool = WeightsThresh.target.isin(a)
# WeightsFiltered = WeightsThresh[WeightsBool]
# print(WeightsFiltered)
# G2 = nx.from_pandas_edgelist(WeightsFiltered)
# print('G2')
# print(G2)
# degree_G2 = nx.degree(G2)
# degree_df = pd.DataFrame(degree_G2, columns = ['Node', 'Degree'])
# print(degree_df)
# degree_freq_G2 = nx.degree_histogram(G2)
# degree_freq_df = pd.DataFrame(degree_freq_G2, columns = ['Frequency'])
# degree_freq_df['Degree'] = degree_freq_df.index
# degree_freq_df = degree_freq_df[['Degree', 'Frequency']]
# degree_df.to_csv('Frequency_Conc_Filt.tsv', sep = '\t')
# nx.draw(G2, with_labels=True)
# plt.show()
# print(degree_freq_df)
# assortativity_G2 = nx.degree_assortativity_coefficient(G2)
# print(assortativity_G2)
