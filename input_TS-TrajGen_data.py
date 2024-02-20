import numpy as np
# import geopandas as gpd
import pandas as pd

split = False
validation_split = .2
shuffle_dataset = True
random_seed= 42

if split:
    df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory.csv')
    dataset_size = len(df_porto)
    indices = list(range(dataset_size))
    
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_data = df_porto.iloc[train_indices]
    train_data.to_csv('Porto-Taxi/Porto_Taxi_trajectory_train.csv')
    
    test_data = df_porto.iloc[val_indices]
    train_data.to_csv('Porto-Taxi/Porto_Taxi_trajectory_test.csv')
    
    
    df_bj=pd.read_csv('BJ-Taxi/BJ_Taxi_201511_trajectory.csv')
    dataset_size = len(df_bj)
    indices = list(range(dataset_size))
    
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_data = df_bj.iloc[train_indices]
    train_data.to_csv('BJ-Taxi/BJ_Taxi_trajectory_train.csv')
    
    test_data = df_bj.iloc[val_indices]
    train_data.to_csv('BJ-Taxi/BJ_Taxi_trajectory_test.csv')
    
else:
    df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_train.csv')
    
    rid_list_list=df_porto.rid_list.values.tolist()
    
    trajectories=[]
    for traj in rid_list_list:
        traj_str=''.join(traj)
        traj_str+='\n'
        trajectories.append(traj_str)
        
    fo = open("TS-TrajGen_Porto.txt", "w")
    for element in trajectories:
        fo.write(element + "\n")
    fo.close()
    
    
    df_bj=pd.read_csv('BJ-Taxi/BJ_Taxi_trajectory_train.csv')
    
    rid_list_list=df_bj.rid_list.values.tolist()
    
    trajectories=[]
    for traj in rid_list_list:
        traj_str=''.join(traj)
        traj_str+='\n'
        trajectories.append(traj_str)
        
    fo = open("TS-TrajGen_BJ.txt", "w")
    for element in trajectories:
        fo.write(element + "\n")
    fo.close()