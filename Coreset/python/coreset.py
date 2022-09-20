import numpy as np
from points import Points
import os
import pandas as pd
import geopandas as gpd
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
def distance(p1, p2):
    return np.sum((p1 - p2) ** 2)


def initial_cluster(data, k):
    '''
    initialized the centers for K-means++
    inputs:
        data - numpy array
        k - number of clusters
    '''
    centers = []
    centers_indices = []
    size = data.shape[0]
    dist = np.zeros(size)
    indices = np.arange(size)
    # plot(data, np.array(centers))

    first_center_id = np.random.choice(indices, 1)[0]
    
    first_center = data[first_center_id]
    

    centers_indices.append(first_center_id)
    centers.append(first_center)

    for nnn in range(k - 1):
        for i in range(size):
            dist[i] = min(distance(c, data[i]) for c in centers)  # Improvement can be done here

        weights = dist / sum(dist)
        print(weights)
        # if weights !=0:
        #     weights = dist / sum(dist)
        # else:
        #     weights = 0.000001

        ## select data point with maximum distance as our next centroid
        next_center_id = np.random.choice(indices, 1, p=weights)[0]

        next_center = data[next_center_id]
        centers_indices.append(next_center_id)
        centers.append(next_center)
        # plot(data, np.array(centers))

    return centers, centers_indices


def _compute_sigma_x(points, centers):
    size = points.shape[0]
    dist = np.zeros(size)
    assign = np.zeros(size, dtype=np.int)  # array to store which center was assigned to each point
    cluster_size = np.zeros(len(centers))  # dict to store how many points in clusters of every center
    for i in range(size):
        cur_dis = np.array([distance(c, points[i]) for c in centers])
        center_id = np.argmin(cur_dis)  # belonged center id for this point
        dist[i] = cur_dis[center_id]
        assign[i] = center_id
        cluster_size[center_id] += 1

    c_apx_x = np.array([cluster_size[c] for c in assign])
    total_sum = dist.sum()
    sigma_x = dist / total_sum + 1 / c_apx_x
    
    return sigma_x


def compute_coreset(points, k, N):
    '''
    Implement the core algorithm of generation of coreset
    :param points:weighted points
    :param k: the amount of initialized centers, caculated by k-means++ method
    :param N: size of coreset
    :return: coreset that generated from points
    '''
    data_size, dimension = points.shape
    assert data_size > N, 'Setting size of coreset is greater or equal to the original data size, please alter it'
    centers, _ = initial_cluster(points, k)
    sigma_x = _compute_sigma_x(points, centers)
    prob_x = sigma_x / sum(sigma_x)
    weights_x = 1 / (N * prob_x)
    samples_idx = np.random.choice(np.arange(data_size), N, p=prob_x)
    samples = np.take(points, samples_idx, axis=0)
    weights = np.take(weights_x, samples_idx, axis=0)
    coreset = Points(N, dimension)
    coreset.fill_points(samples, weights)

    return coreset

def col_data_conversion(data,col):
  data[col] = [x.split(':')[2] for x in data[col]]
  data[col] = pd.to_numeric(data.line_id, errors='coerce').fillna(0).astype(np.int64)



# def data_filter(data):
#     '''
#     Function to filter the original data and discard the none float or integral type columns
#     :param data: input data
#     :return: numerical data after filtering
#     '''
#     numerical_data = data
#     for col in list(data):
#       col_data_conversion(data,col)
    
#     return numerical_data

def data_filter(data):
    '''
    Function to filter the original data and discard the none float or integral type columns
    :param data: input data
    :return: numerical data after filtering
    '''

    numerical_data = data.select_dtypes(include=['float64', 'int64'])
    return numerical_data    

if __name__ == '__main__':
    #data = np.random.randint(0, 100, (10000, 8))
    # detect all the csv files stored inside datafolder
    # read all the CSV files

    path='../../'
    parser = argparse.ArgumentParser(description='File Name')
    parser.add_argument('-f', dest='f', type=str, help='geojson file from which to extract corrdinates from',default='monaco-latest.geojson')
    parser.add_argument('-k', dest='k', type=int, help='the amount of initialized centers, caculated by k-means++ method',default=5)
    parser.add_argument('-n', dest='n', type=int, help='size of coreset',default=50)
    parser.add_argument('-o', dest='o', type=str, help='output destination',default=path)



    args = parser.parse_args()
    f = args.f
    fpath = os.path.join(path,f)
    fname = Path(fpath).stem
    output = path+args.o
    print(output,args.o)
    k = args.k

    n = args.n

    nm_data = gpd.read_file(fpath)
    nm_data["centroid"]=nm_data["geometry"].to_crs(epsg=4326).centroid

    nm_data["lat"]= nm_data["centroid"].x
    # .centroid.map(lambda p: p.x)
    # .centroid.map(lambda p: p.x)
    # .to_crs(epsg=3035)
    nm_data["long"]= nm_data["centroid"].y
    # .centroid.map(lambda p: p.x)
    # .map(lambda p: p.y)
    # .centroid.map(lambda p: p.y)
    # .to_crs(epsg=3035)
    # nm_data["JoinID"]= nm_data.index

    nm_datall=nm_data[["lat","long"]]
    fcsv = fname+'.csv'
    nm_datall.to_csv(fcsv,header=False, index=False)
    # [['lat','lon']]
    data = nm_datall.to_numpy()
    data_csv = pd.read_csv(fcsv)
    nm_data = data_filter(data_csv)
    data = nm_data.to_numpy()
    centers, ids = initial_cluster(data, k)
    coreset = compute_coreset(data, k, n)

    print("centers:",centers,type(centers),"\n")
    print("ids:",ids,type(ids),"\n")
    print("coreset_values:",coreset.get_values(),type(coreset.get_values()),"\n")
    print("coreset_weights:",coreset.get_weights(),type(coreset.get_weights()),"\n")




    df_centers = pd.DataFrame(centers, columns=["lat","long"])
    df_ids = pd.DataFrame(ids, columns=["ids"])

    df_centers.to_csv(output+"/"+'centers.csv', index=False)
    df_ids.to_csv(output+"/"+'ids.csv', index=False)

    np.savetxt(output+"/"+"coreset-values-"+fcsv,coreset.get_values(), delimiter=",")
    np.savetxt(output+"/"+"coreset-weights-"+fcsv,coreset.get_weights(), delimiter=",")