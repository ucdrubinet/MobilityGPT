import numpy as np
import geopandas as gpd
import pandas as pd

df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_train.csv')

rid_list_list=df_porto.rid_list.values.tolist()
# rid_list_list_sub=rid_list_list[:100000]

trajectories=[]
for traj in rid_list_list:
    traj_str=''.join(traj)
    traj_str+='\n'
    trajectories.append(traj_str)
    
fo = open("TS-TrajGen_Porto.txt", "w")
for element in trajectories:
    fo.write(element + "\n")
fo.close()