//
// Created by Xiaobo Wu on 2021/2/26.
//

#include "kmeans.h"
#include "kmeans.cpp"
#include "dataloader.h"
#include "dataloader.cpp"
#include "mr_coreset.cu"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <time.h>

//typedef unsigned long long int size_int;

using namespace std;
using namespace coreset;


int main(int argc, char **argv){
    if (argc != 7) {
    std::cout << "Usage: ./mr_main <csv_file_path> <coreset_size> <cluster_size> <data_dimension> <output_path>  <chunk_size>\n";
    return EXIT_FAILURE;
    }
    
    //Obtain the parameters
    string csv_path = argv[1];
    size_t coreset_size = stoi(argv[2]);
    size_t cluster_size = stoi(argv[3]);
    unsigned int dimension = stoi(argv[4]);
    string output_path = argv[5];
    size_t chunk_size = stoi(argv[6]);
    
    clock_t start,end; //computing the running time
    DataLoader<float> dataloader(dimension); //Create dataloader object

    start = clock();
    
    vector<float> data = dataloader.Loader_1D(csv_path);
    end = clock();
    cout<<"Data loading time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;

    //Running Coreset method:
    size_int n = data.size() / dimension; 
    vector<float> data_weights(n, 1.0);
   
    coreset::FlatPoints coreset;

    start = clock();
    coreset = compute_coreset_mr(data, data_weights, dimension, cluster_size, coreset_size, chunk_size, 0, n -1);
    end = clock();

    cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;

    //Write the output coreset to csv file
    //Get values and weights
    vector<float> v = coreset.GetValues();
    vector<float> w = coreset.GetWeights();
    
    //Generate file output path
    string csv_name = csv_path.substr(csv_path.find_last_of('/') + 1);
    string file_name = csv_name.substr(0, csv_name.find_last_of('.')); //Remove '.csv' to get file name
    string value_path = output_path + file_name + "-coreset_v.csv";
    string weight_path = output_path + file_name + "-coreset_w.csv";
    dataloader.WriteCsv_1D(value_path, v, dimension);
    dataloader.WriteCsv_1D(weight_path, w);

    return 0;
}


