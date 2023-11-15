import pandas as pd
from scipy.spatial import distance
import pickle
from matplotlib import pyplot as plt


df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_test.csv')
samples=df_porto.sample(n=5000)
rid_list_list=samples.rid_list.values.tolist()

porto_geo=pd.read_csv('Porto-Taxi/porto.geo')

OD_test=[]
length_test=[]
for traj in rid_list_list:
    link_ids=list(map(int, traj.split(',')))
    OD_test.append(link_ids[0])
    OD_test.append(link_ids[-1])
    length = porto_geo[porto_geo['geo_id'].isin(link_ids)].length.sum()
    length_test.append(length)
    

file = open('./TS-TrajGen_Porto_synthetic/chargpt_sample/test_trajectories.txt', 'rb')
links = pickle.load(file)

OD_synth=[]
length_synth=[]
for link_ids in links:
    OD_synth.append(link_ids[0])
    OD_synth.append(link_ids[-1])
    length = porto_geo[porto_geo['geo_id'].isin(link_ids)].length.sum()
    length_synth.append(length)

    
js_OD=distance.jensenshannon(OD_synth, OD_test)
js_length=distance.jensenshannon(length_synth, length_test)
    
OD_synth_count=[OD_synth.count(x) for x in set(OD_synth)]
OD_test_count=[OD_test.count(x) for x in set(OD_test)]

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(OD_test_count, label='Raw data', linewidth=6, color='red')
ax.plot(OD_synth_count,'-', label='Synthetic data', linewidth=2, color='blue')
plt.grid(axis='y', alpha=0.75)
# legend = ax.legend(loc='upper right', fontsize='x-large')
legend = ax.legend(fontsize='15')
plt.xticks(fontsize=14)
plt.yticks( fontsize=14)
plt.xlabel('Links', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.title('OD link distribution with adjacency matrix')
plt.show()
