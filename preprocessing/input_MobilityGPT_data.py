import numpy as np
import pandas as pd

split = False
validation_split = .2
shuffle_dataset = True
random_seed= 42

dataset = "SF"

if split:
    df=pd.read_csv(dataset+'-Taxi/'+dataset+'_Taxi_trajectory.csv')
    dataset_size = len(df)
    indices = list(range(dataset_size))
    
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_data = df.iloc[train_indices]
    train_data.to_csv(dataset+'-Taxi/'+dataset+'_Taxi_trajectory_train.csv')
    
    test_data = df.iloc[val_indices]
    test_data.to_csv(dataset+'-Taxi/'+dataset+'_Taxi_trajectory_test.csv')
    
else:
    
    df=pd.read_csv(dataset+'-Taxi/'+dataset+'_Taxi_trajectory_train.csv')
    rid_list_list=df.rid_list.values.tolist()
    
    trajectories=[]
    for traj in rid_list_list:
        traj_str=''.join(traj)
        traj_str+='\n'
        trajectories.append(traj_str)
        
    fo = open("Trajs_"+dataset+".txt", "w")
    for element in trajectories:
        fo.write(element + "\n")
    fo.close()

    
    
