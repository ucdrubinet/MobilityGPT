import pandas as pd
from scipy.spatial import distance
import pickle
from tqdm import tqdm
import numpy as np


df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_test.csv')
samples=df_porto.sample(n=5000)
rid_list=samples.rid_list.values.tolist()

porto_geo=pd.read_csv('Porto-Taxi/porto.geo')
lats = []
lons = []
for ind, geo in porto_geo.iterrows():
    coordinate = list(map(float, geo['coordinates'].replace('[', '').replace(']', '').split(',')))
    lats.append(coordinate[0])
    lons.append(coordinate[1])
gps=[lats, lons]

    

file = open('./TS-TrajGen_Porto_synthetic/chargpt_adj_gravity_sample/test_trajectories.txt', 'rb')
links = pickle.load(file)

OD_synth=[]
length_synth=[]

OD_test=[]
length_test=[]
for i in tqdm(range(len(rid_list))):
    traj=rid_list[i]
    link_ids = links[i]
    if len(link_ids)>0:
        OD_synth.append(link_ids[0])
        OD_synth.append(link_ids[-1])
        length = porto_geo[porto_geo['geo_id'].isin(link_ids)].length.sum()
        length_synth.append(length)

        link_ids=list(map(int, traj.split(','))) 
        OD_test.append(link_ids[0])
        OD_test.append(link_ids[-1])
        length = porto_geo[porto_geo['geo_id'].isin(link_ids)].length.sum()
        length_test.append(length)

def arr_to_distribution(arr, min, max, bins):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution, base[:-1]

OD_synth_dist, _ = arr_to_distribution(OD_synth, min(OD_synth+OD_test), max(OD_synth+OD_test), 300)
OD_test_dist, _ = arr_to_distribution(OD_test, min(OD_synth+OD_test), max(OD_synth+OD_test), 300)

length_synth_dist, _ = arr_to_distribution(length_synth, min(length_synth+length_test), max(length_synth+length_test), 300)
length_test_dist, _ = arr_to_distribution(length_test, min(length_synth+length_test), max(length_synth+length_test), 300)
    
js_OD=distance.jensenshannon(OD_synth_dist, OD_test_dist)
print('JS value for OD: ',js_OD)
js_length=distance.jensenshannon(length_synth_dist, length_test_dist)
print('JS value for length: ',js_length)

    
# OD_synth_count=[OD_synth.count(x) for x in set(OD_synth)]
# OD_test_count=[OD_test.count(x) for x in set(OD_test)]

# fig, ax = plt.subplots(figsize=(10,8))
# ax.plot(OD_test_count, label='Raw data', linewidth=6, color='red')
# ax.plot(OD_synth_count,'-', label='Synthetic data', linewidth=2, color='blue')
# plt.grid(axis='y', alpha=0.75)
# # legend = ax.legend(loc='upper right', fontsize='x-large')
# legend = ax.legend(fontsize='15')
# plt.xticks(fontsize=14)
# plt.yticks( fontsize=14)
# plt.xlabel('Links', fontsize=16)
# plt.ylabel('Frequency', fontsize=16)
# plt.title('OD link distribution with adjacency matrix')
# plt.show()
