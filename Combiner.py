import pandas as pd
import csv
import os

def combine(dirpath, output):
    files =  os.listdir(dirpath)
    num_files = len(files)
    print num_files

    file = dirpath + str(2) + '.csv'
    final_pd = pd.read_csv(file, header=None, names=['Index', '2'])
    column = ['Cluster 2']

    for i in range(1,num_files-1):
        file = dirpath + str(i+2) + '.csv'
        pd_A = pd.read_csv(file, header=None)
        index = str(i+2)
        cluster = pd_A[pd_A.columns[-1]]
        final_pd[index] = cluster
        column.append('Cluster ' + str(i+2))

    # output = bef_dirpath + str(country) + '.csv'
    final_pd = final_pd.drop('Index', axis=1)
    final_pd.columns = column
    final_pd.to_csv(output, sep=',')

def combiner():
    for year in range(2000, 2015):
        intermediate_out = "/home/striker/Factor-Analysis/Output/Revise_Cluster/" + str(year)
        if not os.path.exists(intermediate_out):
            os.makedirs(intermediate_out)

        dirPath = "/home/striker/Factor-Analysis/Output/Clusters/All_Cluster/" + str(year) + "/"
        for country in range(44):
            new_dirPath = dirPath + str(country) + '/'

            outp = "/home/striker/Factor-Analysis/Output/Revise_Cluster/" + str(year) + "/" + str(country) + '.csv'
            combine(new_dirPath, outp)

def silhoutte_combine():
    silhoutte_list = []
    for year in range(2000, 2015):
        file_path = '/home/striker/Factor-Analysis/Results_3/' + str(year) + '/result.csv'
        pd_A = pd.read_csv(file_path)
        silhoutte = pd_A.iloc[:, 13].values
        silhoutte_list.append(silhoutte)
    (pd.DataFrame(silhoutte_list).T).to_csv('/home/striker/Factor-Analysis/Output/Final Results/lpa_dunn.csv', header=False)

if __name__ == '__main__':
    silhoutte_combine()