import numpy as np
import pandas as pd
import csv
import os


def normalize_to_smallest_integers(labels):
    """Normalizes a list of integers so that each number is reduced to the minimum possible integer, maintaining the order of elements.
    :param labels: the list to be normalized
    :returns: a numpy.array with the values normalized as the minimum integers between 0 and the maximum possible value.
    """

    max_v = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
    sorted_labels = np.sort(np.unique(labels))
    unique_labels = range(max_v)
    new_c = np.zeros(len(labels), dtype=np.int32)

    for i, clust in enumerate(sorted_labels):
        new_c[labels == clust] = unique_labels[i]

    return new_c


def dunn(labels, distances):

    labels = normalize_to_smallest_integers(labels)

    unique_cluster_distances = np.unique(min_cluster_distances(labels, distances))
    max_diameter = max(diameter(labels, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def min_cluster_distances(labels, distances):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    """
    labels = normalize_to_smallest_integers(labels)
    n_unique_labels = len(np.unique(labels))

    min_distances = np.zeros((n_unique_labels, n_unique_labels))
    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] != labels[ii] and distances[i, ii] > min_distances[labels[i], labels[ii]]:
                min_distances[labels[i], labels[ii]] = min_distances[labels[ii], labels[i]] = distances[i, ii]
    return min_distances


def diameter(labels, distances):
    """Calculates cluster diameters (the distance between the two farthest data points in a cluster)
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :returns:
    """
    labels = normalize_to_smallest_integers(labels)
    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i + 1, len(labels)):
            if labels[i] == labels[ii] and distances[i, ii] > diameters[labels[i]]:
                diameters[labels[i]] = distances[i, ii]
    return diameters


def find_dunn_index(inde, country):
    rotated_score = ['Dunn Index']
    file_name = '/home/striker/Factor-Analysis/Output/Intermediate_Output/' + str(inde) + '/' + str(country) + '.csv'
    pd_RCM = pd.read_csv(file_name, header=None)
    X = pd_RCM.iloc[:, :-1]
    X = (X.multiply(-1)).add(1)
    cluster_mod = '/home/striker/Factor-Analysis/Clusters_Year_Wise_2/' + str(inde) + '/' + str(country) + '_silh_mod.csv'
    pd_cluster = pd.read_csv(cluster_mod)
    length_tot = len(pd_cluster.iloc[0])

    for index in range(1, length_tot):
        Y = pd_cluster.iloc[:, index].tolist()
        del Y[-1]
        del Y[-1]
        del Y[-1]
        rotated_score.append(round(dunn(Y, euclidean_distances(X)), 8))
    output_file = open('/home/striker/Factor-Analysis/Clusters_Year_Wise_2/' + str(inde) + '/' + str(country) + '_silh_mod_dunn.csv', 'w')
    cluster_file = open(cluster_mod, 'r')
    input_reader = csv.reader(cluster_file, delimiter=',')

    for line in input_reader:
        for val in line:
            output_file.write(val + ',')
        output_file.write('\n')
    for val in rotated_score:
        output_file.write(str(val) + ',')
    return rotated_score


def get_max_silhoute(ind, country,modularity_list, silhoutte_list, cluster_list, louvain_mod, louvain_silh, louvain_clus, nmi, lpa_mod, lpa_silh, lpa_clus, pca_dunn, louvain_dunn, lpa_dunn):
    file_path = '/home/striker/Factor-Analysis/Clusters_Year_Wise_2/' + str(ind) + '/' + str(country) + '_silh_mod_dunn.csv'
    pd_A = pd.read_csv(file_path)
    new_pd_A = pd_A.iloc[:, 1:-1]

    mod = new_pd_A.iloc[56:-3].values
    mod_pca = []
    for index in range((len(mod[0])-3)):
        mod_pca.append(mod[0][index])
    louveen_mod = mod[len(mod)-2]
    lpa_modu = mod[len(mod) - 1]

    silh = new_pd_A.iloc[58:-1].values
    louveen_silh = silh[len(silh)-2]
    lpa_sil = silh[len(silh)- 1]
    nmis = new_pd_A.iloc[57:-2].values

    dun = new_pd_A.iloc[59,:].values
    louveen_dun = dun[len(dun) -3]
    lpa_dun = dun[len(dun) -2]
    dun_pca = []
    for index in range((len(dun) - 3)):
        dun_pca.append(dun[index])

    silh_pca = []
    for index in range((len(silh[0]) - 3)):
        silh_pca.append(silh[0][index])
    max_mod_pcs = max(mod_pca)
    cluster_pca = mod_pca.index(max_mod_pcs)
    silh_cores_pca = silh_pca[cluster_pca]
    nmi_sim = nmis[0][cluster_pca]
    pca_dun = dun_pca[cluster_pca]

    cluster_pca = new_pd_A.iloc[:,cluster_pca].values

    nmi.append(nmi_sim)
    modularity_list.append(max_mod_pcs)
    silhoutte_list.append(silh_cores_pca)
    new_cluster_pca = cluster_pca[:-4]
    cluster_list.append(len(set(new_cluster_pca)))

    loueen_cluster = new_pd_A.iloc[:,-3].values
    new_cluster_louvain = loueen_cluster[:-4]
    louvain_clus.append(len(set(new_cluster_louvain)))
    louvain_mod.append(louveen_mod[len(louveen_mod)-3])
    louvain_silh.append(louveen_silh[len(louveen_silh)-3])

    lpa_mod.append(lpa_modu[len(lpa_modu) -2])
    lpa_silh.append(lpa_sil[len(lpa_sil) -2])
    lpa_cluster = new_pd_A.iloc[:, -2].values
    new_cluster_lpa = lpa_cluster[:-4]
    lpa_clus.append(len(set(new_cluster_lpa)))

    pca_dunn.append(pca_dun)
    louvain_dunn.append(louveen_dun)
    lpa_dunn.append(lpa_dun)

if __name__ == '__main__':
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    # for yea in range(2000,2015):
    #     for coun in range(44):
    #         find_dunn_index(yea, coun)

    direct = '/home/striker/Factor-Analysis/Results_3/'

    for ind in range(2000, 2015):
        if not os.path.exists(direct + str(ind)):
            os.makedirs(direct + str(ind))

        path_dir = '/home/striker/Factor-Analysis/Results_3/' + str(ind) + '/'
        output_file = open(path_dir + 'result.csv', 'w')
        output_file.write(
            'Country Number, Modularity, Silhoutte, PCA Cluster, Louvain Mod, Louvain Silh, Louvain Cluster, NMI, LPA Mod, LPA Silh, LPA Cluster, PCA Dunn, Louvain Dunn, LPA Dunn\n')

        modularity_list = []
        silhoutte_list = []
        cluster_list = []
        louvain_mod = []
        louvain_silh = []
        louvain_clus = []
        nmi = []
        lpa_mod = []
        lpa_silh = []
        lpa_clus = []
        pca_dunn = []
        lpa_dunn = []
        louvain_dunn = []
        for coun in range(44):
            get_max_silhoute(ind, coun, modularity_list, silhoutte_list, cluster_list, louvain_mod, louvain_silh,
                             louvain_clus, nmi, lpa_mod, lpa_silh, lpa_clus, pca_dunn, louvain_dunn, lpa_dunn)
        for index in range(len(modularity_list)):
            output_file.write(
                str(index) + ',' + str(modularity_list[index]) + ',' + str(silhoutte_list[index]) + ',' + str(
                    cluster_list[index]) + ',' + str(louvain_mod[index]) + ',' + str(louvain_silh[index]) + ',' + str(
                    louvain_clus[index]) + ',' + str(nmi[index]) + ',' + str(lpa_mod[index]) + ',' + str(
                    lpa_silh[index]) + ',' + str(lpa_clus[index]) + ',' + str(pca_dunn[index]) + ',' + str(louvain_dunn[index]) + ',' + str(lpa_dunn[index]) + '\n')


