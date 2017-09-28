import csv
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
import factor_rotation as fr
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import heapq

def transform_matrix(filepath):
    input_file = open(filepath, 'r')
    input_reader = csv.reader(input_file, delimiter=',')

    row_sum = [0] * 56
    column_sum = [0] * 56
    row_index = 0

    for row in input_reader:
        for col_index in range(0, len(row)):
            row_sum[row_index] += float(row[col_index])
            column_sum[col_index] += float(row[col_index])
        row_index += 1

    # print row_sum
    # print column_sum

    input_file.seek(0)

    output_A_file = open('/home/striker/Factor-Analysis/Output/A.csv', 'w')
    output_B_file = open('/home/striker/Factor-Analysis/Output/B.csv', 'w')
    output_C_file = open('/home/striker/Factor-Analysis/Output/C.csv', 'w')
    output_D_file = open('/home/striker/Factor-Analysis/Output/D.csv', 'w')

    row_index = 0
    for row in input_reader:
        for col_index in range(0, len(row)):
            value_A = 0
            value_D = 0

            if column_sum[col_index] > 0:
                value_A = float(row[col_index])/column_sum[col_index]
            output_A_file.write(str(value_A) + ',')

            if row_sum[col_index] > 0:
                     value_D = float(row[col_index])/ row_sum[col_index]
            output_D_file.write(str(value_D) + ',')

            value_B = 0
            if column_sum[row_index] > 0:
                value_B = float(row[col_index])/ column_sum[row_index]
            output_B_file.write(str(value_B) + ',')

            value_C = 0
            if row_sum[row_index] > 0:
                value_C = float(row[col_index]) / row_sum[row_index]

            output_C_file.write(str(value_C) + ',')

        output_A_file.write("\n")
        output_D_file.write("\n")
        output_B_file.write("\n")
        output_C_file.write("\n")
        row_index += 1

def correlation_matrix(msize, index, countr):
    pd_A = pd.read_csv('/home/striker/Factor-Analysis/Output/A.csv', header=None)
    pd_B = pd.read_csv('/home/striker/Factor-Analysis/Output/B.csv', header=None)
    pd_C = pd.read_csv('/home/striker/Factor-Analysis/Output/C.csv', header=None)
    pd_D = pd.read_csv('/home/striker/Factor-Analysis/Output/D.csv', header=None)
    result = open('/home/striker/Factor-Analysis/Output/Intermediate_Output/' + str(index) + '/' + str(countr) + '.csv','w')

    AB = [[0]*msize for i in range(msize)]
    BC = [[0]*msize for i in range(msize)]
    AD = [[0]*msize for i in range(msize)]
    CD = [[0]*msize for i in range(msize)]

    for row_i in range(0, msize):
        for col_i  in range(0, msize):
            Ai = pd_A[row_i].values
            Bi = pd_B[col_i].values

            if math.isnan(np.corrcoef(Ai, Bi)[0][1]):
                AB[row_i][col_i] = 0
            else:
                AB[row_i][col_i] = np.corrcoef(Ai, Bi)[0][1]

            Di = pd_D[col_i].values
            if math.isnan(np.corrcoef(Ai, Di)[0][1]):
                AD[row_i][col_i] = 0
            else:
                AD[row_i][col_i] = np.corrcoef(Ai, Di)[0][1]

            BTi = pd_B[row_i].values
            Ci = pd_C[col_i].values
            if math.isnan(np.corrcoef(BTi, Ci)[0][1]):
                BC[row_i][col_i] = 0
            else:
                BC[row_i][col_i] = np.corrcoef(BTi, Ci)[0][1]

            CTi = pd_C[row_i].values
            if math.isnan(np.corrcoef(CTi, Di)[0][1]):
                CD[row_i][col_i] = 0
            else:
                CD[row_i][col_i] = np.corrcoef(CTi, Di)[0][1]

    pre_temp = [[0]*msize for i in range(msize)]
    for row_i in range(msize):
        for col_i in range(msize):
            pre_temp[row_i][col_i] = max(AB[row_i][col_i], BC[row_i][col_i], AD[row_i][col_i], CD[row_i][col_i])
            # print pre_temp[row_i][col_i]

    final_matrix = [[0]*msize for i in range(msize)]
    for row_i in range(msize):
        for col_i in range(msize):
            final_matrix[row_i][col_i] = abs(max(pre_temp[row_i][col_i], pre_temp[col_i][row_i]))
            result.write(str(final_matrix[row_i][col_i]) + ',')
        result.write("\n")

def PCA(inde, countr):
    pd_A = pd.read_csv( '/home/striker/Factor-Analysis/Output/Intermediate_Output/' + str(inde) + '/' + str(countr) + '.csv', header=None)
    pd_A = pd_A.iloc[:, :-1]
    temp_pd_A = pd_A

    pd_A = pd_A.loc[(pd_A != 0).any(axis=0), :]
    pd_A = pd_A.loc[:, (pd_A != 0).any(axis=0)]

    X_std = StandardScaler().fit_transform(pd_A)
    cor_mat = np.corrcoef(X_std.T)

    eigenvalues = np.linalg.eigvals(cor_mat)
    _eigenvectors = np.linalg.eig(cor_mat)[1]
    eigenvectors =  _eigenvectors * np.sign(np.sum(_eigenvectors, 0))

    new_eig_vectors =  eigenvectors * pow(eigenvalues, .5)
    eig_pairs = [(np.abs(eigenvalues[i]), new_eig_vectors[:, i]) for i in range(len(eigenvalues))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    new_eig_pairs = []
    for i in eig_pairs:
        if i[0] > 1:
            new_eig_pairs.append(i[1].tolist())
    new_eig_pairs = zip(*new_eig_pairs)
    new_eig_pairs = np.array(new_eig_pairs)

    for cluster in range(2, len(new_eig_pairs[0])+1):
        clusterred_eig_pairs = new_eig_pairs[:, :cluster]
        L, T = fr.rotate_factors(clusterred_eig_pairs, 'varimax')

        file_name = '/home/striker/Factor-Analysis/Output/Clusters/RCM/' + str(inde)+ '/' +  str(countr) + '/' + str(cluster) + '.csv'
        rotated_matrix  = open(file_name, 'w')

        file_all_cluster = '/home/striker/Factor-Analysis/Output/Clusters/All_Cluster/' + str(inde)+ '/' +  str(countr) + '/' + str(cluster) + '.csv'
        clusters_num = open(file_all_cluster, 'w')
        clusters = []
        for row_i in range(len(L)):
            first_qtr = 0
            second_qtr = 0
            third_qtr = 0

            for col_i in range(len(L[0])):
                value = abs(L[row_i][col_i])

                if value > 0.65:
                    first_qtr = col_i + 1
                elif 0.5 < value < 0.65:
                    second_qtr = col_i + 1
                elif 0.35 < value < 0.5:
                    third_qtr = col_i + 1

                rotated_matrix.write(str(round(L[row_i][col_i], 5)) + ',')

            if first_qtr == 0 and second_qtr != 0:
                first_qtr = second_qtr
            elif first_qtr == 0 and third_qtr != 0:
                first_qtr = third_qtr

            rotated_matrix.write(str(first_qtr) + '\n')
            clusters.append(str(first_qtr))

        cluster_index  = 0

        for ind in range(0, 56):
            file_val = temp_pd_A[ind].sum()
            if float(file_val) == float(0.0):
                clusters_num.write(str(ind+1) + ',' + str(99) + '\n')
            else:
                clusters_num.write(str(ind+1) + ',' + str(clusters[cluster_index]) + '\n')
                cluster_index += 1


    return len(new_eig_pairs[0])+1

def find_silhoutte_score(inde, lk, country):
    rotated_score = []
    for cluster in range(2, lk):
        file_name = '/home/striker/Factor-Analysis/Output/Clusters/RCM/' + str(inde) + '/' +  str(country)+ '/' +  str(cluster) + '.csv'
        pd_RCM = pd.read_csv(file_name, header=None)
        X = pd_RCM.iloc[:, :-1]
        Y = pd_RCM.iloc[:, -1]
        rotated_score.append(round(silhouette_score(X, Y, metric='euclidean'), 3))
    return rotated_score

def create_file(filename):
    pd_2014 = pd.read_csv('/home/striker/Factor-Analysis/Input/' + str(filename) + '.csv')
    coli = 6
    rowi = 0
    for i in range(44):
        print i
        reader = pd_2014.iloc[rowi:rowi + 56, coli:coli + 56]
        rowi += 56
        coli += 56
        file_name = '/home/striker/Factor-Analysis/Input/' + str(filename) + '/' + str(i) + '.csv'
        output = open(file_name, 'w')
        for index, row in reader.iterrows():
            count = 0
            for value in row.values:
                output.write(str(value))
                if count < 55:
                    output.write(',')
                count += 1
            output.write("\n")

def create_heatmap():
    pd_2014 = pd.read_csv('/home/striker/Factor-Analysis/Input/' + str(2000) + '.csv')
    country = (reversed(np.unique(pd_2014['Country'].values)))
    y1 = []
    for i in range(0,45):
        y1.append(i)

    fig, ax1 = plt.subplots()
    pd_a = pd.read_csv('/home/striker/Factor-Analysis/Output/Final Results/abs_silhoutte_score_first.csv', header=None)
    data = pd_a.values.tolist()

    sns.heatmap(data, cmap="YlGnBu", linewidths=2.5)
    x1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    squad = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014]

    ax1.set_xticks(x1)
    ax1.set_xticklabels(squad, minor=False, rotation=45)
    plt.xlabel('Year')

    ax1.set_yticks(y1)
    ax1.set_yticklabels(country, minor=False, rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Country')
    plt.show()

    plt.savefig('/home/striker/Factor-Analysis/Output/Abs_Silhoutte_First_Max.png')
    plt.close()

if __name__ == '__main__':
    cluster_year_wise = '/home/striker/Factor-Analysis/Clusters_Year_Wise/'

    for index in range(2000,2015):
        print index

        cluster_year_wise = cluster_year_wise + str(index) + '/'

        direct = '/home/striker/Factor-Analysis/Input/' + str(index)
        if not os.path.exists(direct):
            os.makedirs(direct)

        intermediate_out = '/home/striker/Factor-Analysis/Output/Intermediate_Output/' + str(index)
        if not os.path.exists(intermediate_out):
            os.makedirs(intermediate_out)

        rcm_dir = '/home/striker/Factor-Analysis/Output/Clusters/RCM/' + str(index) + '/'
        cluster_dir = '/home/striker/Factor-Analysis/Output/Clusters/All_Cluster/' + str(index) + '/'
        for i in range(0, 44):
            if not os.path.exists(rcm_dir + str(i)):
                os.makedirs(rcm_dir + str(i))

            if not os.path.exists(cluster_dir + str(i)):
                os.makedirs(cluster_dir + str(i))

        # sil_dir = '/home/striker/Factor-Analysis/Output/Silhoutte/' + str(index) + '/'
        # for i in range(0, 45):
        #     if not os.path.exists(sil_dir + str(i)):
        #         os.makedirs(sil_dir + str(i))

        create_file(index)

        output = open('/home/striker/Factor-Analysis/Output/Clusters/Cluster_' + str(index) + '.csv', 'w')
        outputs = open('/home/striker/Factor-Analysis/Output/Silhoutte/Silhoutte_' + str(index) +' .csv', 'w')

        output.write('Country, Cluster \n')
        outputs.write('Country, Silhoutte \n')
        for i in range(44):
            transform_matrix('/home/striker/Factor-Analysis/Input/' + str(index) + '/' + str(i) + '.csv')
            correlation_matrix(56, index, i)
            lk = PCA(index, i)
            score_include =  find_silhoutte_score(index, lk, i)

            # cluster_year_wise = cluster_year_wise + str(i) + '_mod.csv'
            # sikl_cluster_year_wise = cluster_year_wise + str(i) + '_sill_mod.csv'
            #
            # pd_cluster_year_wise = pd.read_csv(cluster_year_wise)
            # max_value = max(score_include)
            # new_score = ['Silhoutte']
            # for score in score_include:
            #     new_score.append(str(score))
            seconda_max = heapq.nlargest(2, score_include)[0]
            # del new_score[-1]

            max_index = score_include.index(seconda_max)

            output.write(str(i) + ',' + str(max_index+1) + '\n')
            outputs.write(str(i) + ',' + str(seconda_max) + '\n')

            print 'For country: ', i , score_include
            x_axis = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    create_heatmap()