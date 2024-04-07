import pickle
import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
# import geopandas as gpd
# from shapely.geometry import Point, LineString

dataset = "Porto"

porto_rel = pd.read_csv(dataset+'-Taxi/roadmap.rel')
graph = nx.from_pandas_edgelist(porto_rel, source='origin_id', target='destination_id')

file = open(dataset+'-Taxi/regions','rb')
regions, region_links, links_region = pickle.load(file)


# Calculate sampling probabilities based on the number of elements in each key
total_elements = sum(len(values) for values in region_links.values())
sampling_probabilities = {key: len(values) / total_elements for key, values in region_links.items()}

num_trajs = int(1e6)
fo = open("TS-TrajGen_"+dataset+"_random.txt", "w")
for i in tqdm(range(num_trajs)):
    # Sample two keys with replacement based on their probabilities
    sampled_keys = random.choices(list(region_links.keys()), weights=list(sampling_probabilities.values()), k=2)
    
    # Sample one element from each of the sampled keys
    sampled_elements = [random.choice(region_links[key]) for key in sampled_keys]
    traj = nx.shortest_path(G=graph, source=sampled_elements[0], target=sampled_elements[1]) 
    traj_str=','.join([str(t) for t in traj])
    traj_str+='\n'
    fo.write(traj_str + "\n")
fo.close()
    