import pandas as pd
from sklearn.metrics import silhouette_score
import csv
import os
#
# rotated_score = []
# for index in range(2, 14):
#     filename = '/home/striker/MData/Input/' + str(index) + '.csv'
#     L = pd.read_csv(filename, header= None).values
#
#     Y = []
#     for row_i in range(len(L)):
#         first_qtr = 0
#         second_qtr = 0
#         third_qtr = 0
#
#         for col_i in range(len(L[0])):
#             value = abs(L[row_i][col_i])
#
#             if value > 0.65:
#                 first_qtr = col_i + 1
#             elif 0.5 < value < 0.65:
#                 second_qtr = col_i + 1
#             elif 0.35 < value < 0.5:
#                 third_qtr = col_i + 1
#
#         if first_qtr == 0 and second_qtr != 0:
#             first_qtr = second_qtr
#         elif first_qtr == 0 and third_qtr != 0:
#             first_qtr = third_qtr
#         Y.append(first_qtr)
#     print (round(silhouette_score(L, Y, metric='euclidean'), 3))
# # print rotated_score


def find_silhoutte_score(inde, country):
    rotated_score = ['Silhoutte']
    file_name = '/home/striker/Factor-Analysis/Output/Intermediate_Output/' + str(inde) + '/' + str(country) + '.csv'
    pd_RCM = pd.read_csv(file_name, header=None)
    X = pd_RCM.iloc[:, :-1]
    X = (X.multiply(-1)).add(1)
    cluster_mod = '/home/striker/Factor-Analysis/Clusters_Year_Wise_2/' + str(inde) + '/' + str(country) + '_mod.csv'
    pd_cluster = pd.read_csv(cluster_mod)
    length_tot = len(pd_cluster.iloc[0])

    for index in range(1, length_tot):
        Y = pd_cluster.iloc[:, index].tolist()
        del Y[-1]
        del Y[-1]
        rotated_score.append(round(silhouette_score(X, Y, metric='euclidean'), 8))
    output_file = open('/home/striker/Factor-Analysis/Clusters_Year_Wise_2/' + str(inde) + '/' + str(country) + '_silh_mod.csv', 'w')
    cluster_file = open(cluster_mod, 'r')
    input_reader = csv.reader(cluster_file, delimiter=',')

    for line in input_reader:
        for val in line:
            output_file.write(val + ',')
        output_file.write('\n')
    for val in rotated_score:
        output_file.write(str(val) + ',')
    return rotated_score

def get_max_silhoute(ind, country,modularity_list, silhoutte_list, cluster_list, louvain_mod, louvain_silh, louvain_clus, nmi, lpa_mod, lpa_silh, lpa_clus):
    file_path = '/home/striker/Factor-Analysis/Clusters_Year_Wise_2/' + str(ind) + '/' + str(country) + '_silh_mod.csv'
    pd_A = pd.read_csv(file_path)
    new_pd_A = pd_A.iloc[:, 1:-1]

    mod = new_pd_A.iloc[56:-2].values
    mod_pca = []
    for index in range((len(mod[0])-2)):
        mod_pca.append(mod[0][index])
    louveen_mod = mod[len(mod)-2]
    lpa_modu = mod[len(mod) - 1]

    silh = new_pd_A.iloc[58:].values
    louveen_silh = silh[len(silh)-2]
    lpa_sil = silh[len(silh)- 1]
    nmis = new_pd_A.iloc[57:-1].values


    silh_pca = []
    for index in range((len(silh[0]) - 2)):
        silh_pca.append(silh[0][index])
    max_mod_pcs = max(mod_pca)
    cluster_pca = mod_pca.index(max_mod_pcs)
    silh_cores_pca = silh_pca[cluster_pca]
    nmi_sim = nmis[0][cluster_pca]
    cluster_pca = new_pd_A.iloc[:,cluster_pca].values

    nmi.append(nmi_sim)
    modularity_list.append(max_mod_pcs)
    silhoutte_list.append(silh_cores_pca)
    new_cluster_pca = cluster_pca[:-3]
    cluster_list.append(len(set(new_cluster_pca)))

    loueen_cluster = new_pd_A.iloc[:,-2].values
    new_cluster_louvain = loueen_cluster[:-3]
    louvain_clus.append(len(set(new_cluster_louvain)))
    louvain_mod.append(louveen_mod[len(louveen_mod)-2])
    louvain_silh.append(louveen_silh[len(louveen_silh)-2])

    lpa_mod.append(lpa_modu[len(lpa_modu) -1])
    lpa_silh.append(lpa_sil[len(lpa_sil) -1])
    lpa_cluster = new_pd_A.iloc[:, -1].values
    new_cluster_lpa = lpa_cluster[:-3]
    lpa_clus.append(len(set(new_cluster_lpa)))

if __name__ == '__main__':
    direct = '/home/striker/Factor-Analysis/Results_2/'

    for ind in range(2000,2015):
        if not os.path.exists(direct +  str(ind)):
            os.makedirs(direct + str(ind))

        path_dir = '/home/striker/Factor-Analysis/Results_2/' + str(ind) + '/'
        output_file = open(path_dir + 'result.csv', 'w')
        output_file.write('Country Number, Modularity, Silhoutte, PCA Cluster, Louvain Mod, Louvain Silh, Louvain Cluster, NMI, LPA Mod, LPA Silh, LPA Cluster\n')

        modularity_list  = []
        silhoutte_list = []
        cluster_list = []
        louvain_mod = []
        louvain_silh = []
        louvain_clus = []
        nmi = []
        lpa_mod = []
        lpa_silh = []
        lpa_clus = []
        for coun in range(44):
            get_max_silhoute(ind, coun, modularity_list, silhoutte_list, cluster_list, louvain_mod, louvain_silh, louvain_clus, nmi, lpa_mod, lpa_silh, lpa_clus)
        for index in  range(len(modularity_list)):
            output_file.write(str(index) + ',' + str(modularity_list[index]) + ',' + str(silhoutte_list[index]) + ',' + str(cluster_list[index]) + ',' + str(louvain_mod[index]) + ',' + str(louvain_silh[index]) + ',' + str(louvain_clus[index]) + ',' + str(nmi[index]) + ','+ str(lpa_mod[index]) + ','+  str(lpa_silh[index])+  ',' + str(lpa_clus[index])+ '\n')
    # for yea in range(2000,2015):
    #     for coun in range(44):
    #         find_silhoutte_score(yea, coun)