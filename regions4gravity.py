import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, LineString
import pyproj
from shapely.ops import transform
from functools import partial
import geopy.distance
import pickle

def convert_gps_to_2d_grid(gps_locations, lines, distance):
    """Converts GPS locations to 2D grids.
      
    Args:
      gps_locations: A list of GPS locations, each of which is a tuple of (latitude, longitude).
      grid_size: The size of the 2D grid.
      
    Returns:
      A 2D grid of GPS locations.
    """
      
    # Get the minimum and maximum latitudes and longitudes.
    min_latitude = min(gps_locations[1])
    max_latitude = max(gps_locations[1])
    min_longitude = min(gps_locations[0])
    max_longitude = max(gps_locations[0])
    

    line1 = LineString([(min_longitude, min_latitude), (min_longitude, max_latitude)])
    line2 = LineString([(min_longitude, min_latitude), (max_longitude, min_latitude)])
    
    # Geometry transform function based on pyproj.transform
    project = partial(
        pyproj.transform,
        pyproj.Proj('EPSG:4326'),
        pyproj.Proj('EPSG:32633'))
    
    line_lat = transform(project, line1)
    line_lon = transform(project, line2)
    grid_lat= int(line_lat.length//distance)
    grid_lon= int(line_lon.length//distance)
      
    # Create a 2D grid of coordinates.
    grid_coordinates = [np.linspace(min_latitude, max_latitude, grid_lat), np.linspace(min_longitude, max_longitude, grid_lon)]
    
    def square_poly(lat, lon, distance=distance):
          gs = gpd.GeoSeries(wkt.loads(f'POINT ({lon} {lat})'))
          gdf = gpd.GeoDataFrame(geometry=gs)
          gdf.crs='EPSG:4326'
          gdf = gdf.to_crs('EPSG:3857')
          res = gdf.buffer(
              distance=distance,
              cap_style=3)
      
          return res.to_crs('EPSG:4326').iloc[0]
        
    # Convert the GPS locations to grid coordinates.
    grid_polies = []
    center_coords = []
    ids = []
    # links_all=[]
    idd=0
    region_links={}
    links_region={}
    for i in range(grid_lat):
        for j in range(grid_lon):
            poly = square_poly(grid_coordinates[0][i], grid_coordinates[1][j])
            grid_polies.append(poly)
            center_coords.append([grid_coordinates[0][i], grid_coordinates[1][j]])
            ids.append(idd)
            links = []
            for k in range(len(gps_locations[0])):
                if lines[k].intersects(poly):
                    links.append(k) 
                    links_region[k] = idd
            # links_all.append(','.join(str(l) for l in links))
            region_links[idd]=links
            idd+=1
                    
                
    d = {'region_id': ids, 'center':center_coords, 'geometry': grid_polies}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326") 
    
    return gdf, region_links, links_region

porto_geo=pd.read_csv('./Porto-Taxi/porto.geo')   
lats = []
lons = []
coords = []
for ind, geo in porto_geo.iterrows():
    coordinate = list(map(float, geo['coordinates'].replace('[', '').replace(']', '').split(',')))
    coords.append(LineString([[coordinate[i], coordinate[i+1]] for i in range(0, len(coordinate), 2)]))
    lats.append(coordinate[0])
    lons.append(coordinate[1])
gps=[lats, lons]


regions, region_links, links_region = convert_gps_to_2d_grid(gps, coords, 1000)

file = open('Porto-Taxi/regions','wb')
pickle.dump([regions, region_links, links_region],file)

df_porto=pd.read_csv('Porto-Taxi/Porto_Taxi_trajectory_train.csv')
regions_count=np.zeros((2, len(regions)))
for idx, traj in df_porto.iterrows():
    links = list(map(int, traj.rid_list.split(',')))
    for r in range(len(regions)):
        if links[0] in region_links[r]:
            regions_count[0][r]+=1
        if links[-1] in region_links[r]:
            regions_count[1][r]+=1
            
regions_count=regions_count.sum(axis=0, keepdims=True)

gravity=np.zeros((len(regions), len(regions)))
for r1 in range(len(regions)):
    for r2 in range(len(regions)):
        dist=geopy.distance.geodesic(regions.center.iloc[r1], regions.center.iloc[r2]).m
        if dist!=0:
            gravity[r1, r2]=regions_count[0, r1]*regions_count[0, r2]/(dist**2)
            
# np.save('Porto-Taxi/gravity_1000.npy', gravity)   

gravity_traj = [] 
for idx, traj in df_porto.iterrows():
    links = list(map(int, traj.rid_list.split(',')))
    r_o = links_region[links[0]]
    r_d = links_region[links[-1]]
    gravity_traj.append(gravity[r_o, r_d])

df_porto['gravity'] = gravity_traj

df_porto.to_csv('Porto-Taxi/Porto_Taxi_trajectory_train_w_gravity.csv')

    