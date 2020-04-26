
import random
from mod_deepwalk.graph import Graph
import networkx as nx
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


G = nx.read_weighted_edgelist('mp_tweets_mentions.edgelist')
for i in G['F_ElKhoury']:
    print(i)
list(G['F_ElKhoury'].keys())

current_node = G['F_ElKhoury']
transitions = list(current_node.keys())
edge_weights = [current_node[other_node]['weight'] for other_node in transitions]


nx.classes.coreviews.AtlasView

['FP_Champagne']['weight']
rand = random.Random()

rand.choice()


rand = np.random
rand.choice()

np.random.Random()


G = Graph()
with open('mp_tweets_mentions.edgelist') as f:
    for l in f:
        line = l.strip().split()
        key = line[0]
        value = {line[1]: {'weight': line[2]}}
        G[key].update(value)


deepwalk_output = dict()
with open('test_output', 'rb') as f:
    counter = 0
    for line in f:
        if counter == 0:
            counter += 1
            pass
        else:
            mp = line.strip().split()
            key = mp[0]
            value = mp[1:]
            deepwalk_output[key] = value


deepwalk_df = pd.DataFrame(deepwalk_output).T

deepwalk_df_test = deepwalk_df.loc[[index for index in deepwalk_df.index if index != b'MinJusticeEn']]


pca = PCA(n_components=2)

dimensionality_reduce = pca.fit_transform(deepwalk_df_test)

kmeans = KMeans(n_clusters=6)
model = kmeans.fit(dimensionality_reduce)
cluster_prediction = model.predict(dimensionality_reduce)

deepwalk_df_test['prediction'] = cluster_prediction
deepwalk_df_test[deepwalk_df_test.prediction == 5]


plt.scatter(dimensionality_reduce[:, 0], dimensionality_reduce[:, 1], c= cluster_prediction)

mp_of_interest = [b'KamalKheraLib', b'SoniaLiberal', b'MPRubySahota',
                  b'RajLiberal', b'gagansikand', b'Puglaas',
                  b'JustinTrudeau', b'iamIqraKhalid', b'NavdeepSBains',
                  b'sukhdhaliwal', b'HonAhmedHussen', b'HonAhmedHussen',
                  b'Yasmin_Ratansi', b'RajSainiMP', b'SeamusORegan', b'NickWhalenMP',
                  b'avalonMPKen', b'YRobillardPLC', b'daviddbgraham', b'HarjitSajjan',
                  b'mary_ng', b'VoteSheehan', b'MPJatiSidhu', b'pierrebretonplc', b'MPMihychuk',
                  b'VBadawey', b'Kevin_Lamoureux', b'melaniejoly', b'Bill_Morneau', b'TerryDuguid']

mp_real_name = {
    b'KamalKheraLib': 'Kamal Khera',
    b'MPRubySahota': 'Ruby Sahota',
    b'JustinTrudeau': 'Justin Trudeau',
    b'HonAhmedHussen': 'Ahmed Hussen',
    b'NickWhalenMP': 'Nick Whalen',
    b'SeamusORegan': 'Seamus ORegan',
    b'HarjitSajjan': 'Harjit Sajjan',
    b'Kevin_Lamoureux': 'Kevin Lamoureux',
    b'MPMihychuk': 'MaryAnn Mihcyuk',
    b'melaniejoly': 'Melanie Joly',
    b'Bill_Morneau': 'Bill Morneau',
    b'gagansikand': 'Gagan Sikand',
    b'avalonMPKen': 'Ken McDonald',
    b'TerryDuguid': 'Terry Duguid',
}

for mp in mp_real_name:
    mp_index = np.where(deepwalk_df_test.index == mp)
    plt.annotate(mp_real_name[mp], (dimensionality_reduce[mp_index, 0], dimensionality_reduce[mp_index, 1]))

plt.close()

max(dimensionality_reduce[:, 0])

deepwalk_df

final_result = 189
thing_to_sum = 189
for i in range(thing_to_sum):
    thing_to_sum -= 1
    final_result += thing_to_sum