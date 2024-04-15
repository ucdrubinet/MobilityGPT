import pickle
import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, LineString
import osmnx as ox
crs = {'init': 'epsg:4326'} 

def save_to_gdf(traj, edges):
    df=edges.loc[edges['geo_id'].isin(traj)].reset_index(drop=True)
    coordinates = [list(map(float, c.replace('[', '').replace(']', '').split(','))) for c in df['coordinates'].tolist()]
    

    geometry=[]
    for x in coordinates:
        geometry.append(LineString([Point(pair) for pair in zip(x[::2], x[1::2])]))
    df['geometry'] = geometry

    df['name'] = str(i)      
    df=df[['geo_id', 'name', 'coordinates', 'length', 'geometry']]
    gdf = gpd.GeoDataFrame(df, geometry="geometry",  crs = crs)

    fdf = ox.project_gdf(gdf,to_latlong=True)
    fdf.to_file('random_'+str(i)+'_.geojson',driver='GeoJSON')
    

dataset = "SF"

rel = pd.read_csv(dataset+'-Taxi/roadmap.rel')
graph = nx.from_pandas_edgelist(rel, source='origin_id', target='destination_id')

file = open(dataset+'-Taxi/regions','rb')
regions, region_links, links_region = pickle.load(file)

edges = pd.read_csv(dataset+'-Taxi/roadmap.geo')

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
    # save_to_gdf(traj, edges)

    fo.write(traj_str + "\n")
fo.close()
    