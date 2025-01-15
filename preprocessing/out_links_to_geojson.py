# from scipy.stats import gamma, multivariate_normal
# from random import sample
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
import shapely as shp
import random
from itertools import groupby
import time
# import sqlalchemy
import utm
import pickle
from shapely.geometry import LineString, Point


edges =pd.read_csv('/media/ahaydari/2TB_extra/porto_dataset/Porto-Taxi/porto.geo')
# df_porto=pd.read_csv('/media/ahaydari/2TB_extra/porto_dataset/Porto-Taxi/Porto_Taxi_trajectory.csv')

# file = open('./outTS-TrajGen_Porto/chargpt/test_trajectories.txt', 'rb')
links = open('TS-TrajGen_Porto_random.txt', 'r').read()
# links = pickle.load(file)

crs = {'init': 'epsg:4326'} 

links_df=[]
i=1
for link in links:
    index_list = []
    for p in link:
        index = p
        if index not in index_list:
            index_list.append(int(index))

    df=edges.loc[edges['geo_id'].isin(index_list)].reset_index(drop=True)
    coordinates = [list(map(float, c.replace('[', '').replace(']', '').split(','))) for c in df['coordinates'].tolist()]

    geometry=[]
    for x in coordinates:
        geometry.append(LineString([Point(pair) for pair in zip(x[::2], x[1::2])]))
    df['geometry'] = geometry

    df['name'] = str(i)      
    df=df[['geo_id', 'name', 'coordinates', 'length', 'geometry']]
    gdf = gpd.GeoDataFrame(df, geometry="geometry",  crs = crs)

    fdf = ox.project_gdf(gdf,to_latlong=True)
    fdf.to_file('./outTS-TrajGen_Porto/chargpt/'+str(i)+'.geojson',driver='GeoJSON')

    i+=1
    links_df.append(df)
    if i==100: break
