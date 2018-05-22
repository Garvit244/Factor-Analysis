import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns

def twoAxisPlot():
    # Get the values into x, y1, and y2.
    pd_A = pd.read_csv('/home/striker/Factor-Analysis/Results_3/2014/result.csv')
    mod = pd_A.iloc[:,11]
    silh = pd_A.iloc[:,12]
    lpa = pd_A.iloc[:, 13]
    x = pd_A.iloc[:,0]
    nmi = pd_A.iloc[:, 7]

    # Plot y1 vs x in blue on the left vertical axis.
    plt.xlabel("Country")
    plt.ylabel("Modularity")
    # plt.tick_params(axis="y", labelcolor='C1')
    plt.plot(x, mod, "C1-", linewidth=2, label='PCA')
    # plt.plot(x, lpa, "b-", linewidth=2, label='LPA')

    # plt.plot(x, silh, "g-", linewidth=2, label='Louvain')
    # plt.yticks(np.arange(0,13, 1))

    # Plot y2 vs x in red on the right vertical axis.
    # plt.twinx()
    # plt.ylabel("Silhoutte Louvain/ LPA", color="g")
    # plt.tick_params(axis="y", labelcolor="g")
    plt.plot(x, silh, "b-", linewidth=2, label='Louvain')
    plt.plot(x, lpa, "r-", linewidth=2, label='LPA')
    # plt.yticks(np.arange(0,6,1))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)
    (plt.axes()).grid(False)
    # plt.show()
    # plt.gca().invert_yaxis()
    plt.savefig("/home/striker/Factor-Analysis/Results_2/2014/Modularity.png", dpi=150, format="png")
    plt.close()


def dplot():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(1, 10, 0.25)
    Y = np.arange(1, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def create_heatmap():
    # country = reversed(['AUS', 'AUT', 'BEL', 'BGR', 'BRA', 'CAN', 'CHE', 'CHN', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IDN', 'IND', 'IRL', 'ITA', 'JPN', 'KOR', 'LTU', 'LUX', 'LVA', 'MEX', 'MLT', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SVK', 'SVN', 'SWE', 'TUR', 'TWN', 'USA', 'ROW',''])
    y1 = []
    for i in range(0,45):
        y1.append(i)

    fig, ax1 = plt.subplots()
    pd_a = pd.read_csv('/home/striker/Factor-Analysis/Output/Final Results/pca_dunn_country_wise.csv', header=None)
    pd_a = pd_a.iloc[:, 1:]
    data = (pd_a.iloc[:,:15]).values.tolist()
    country_name = pd_a.iloc[:,16].values.tolist()
    country_type = pd_a.iloc[:,15].values.tolist()
    country_name_type = []
    for index in range(0, len(country_name)):
        country_name_type.append(country_type[index] + '_' + country_name[index])
    country = reversed(country_name_type)


    sns.heatmap(data, cmap="YlGnBu", linewidths=2)
    x1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    squad = [' 2000','  2001',' 2002',' 2003',' 2004',' 2005',' 2006',' 2007',' 2008',' 2009',' 2010',' 2011',' 2012',' 2013',' 2014']
    ax1.set_xticks(x1)
    ax1.set_xticklabels(squad, minor=False, rotation='horizontal')
    plt.xlabel('Year')

    ax1.set_yticks(y1)
    ax1.set_yticklabels(country, minor=False, rotation='horizontal')
    plt.xlabel('Year')
    plt.ylabel('Country')
    plt.show()

    plt.savefig('/home/striker/Factor-Analysis/Output/Silhoutte.png')
    plt.close()

def create_new_heatmap(year_index):
    y1 = []
    for i in range(0, 43):
        y1.append(i)

    fig, ax1 = plt.subplots()
    pd_pca = pd.read_csv('/home/striker/Downloads/Processed Final Result/PCA_Cluster.csv', header=None)
    pd_lpa = pd.read_csv('/home/striker/Downloads/Processed Final Result/LPA_Clus.csv', header=None)
    pd_lou = pd.read_csv('/home/striker/Downloads/Processed Final Result/Louvain_Clus.csv', header=None)
    pd_fuzzy = pd.read_csv('/home/striker/Downloads/Processed Final Result/Fuzzy_Clus.csv', header=None)
    pd_index = pd.read_csv('/home/striker/Downloads/Processed Final Result/New_order_by_countries.csv')

    new_pd = pd.DataFrame()
    new_pd["PCA"], new_pd["LPA"], new_pd["Louvain"], new_pd["Fuzzy"] = pd_pca[year_index], pd_lpa[year_index], pd_lou[year_index], pd_fuzzy[year_index]
    new_pd["Country"], new_pd["index"] = pd_index["country_code"], pd_index["index"]
    new_pd["Region"] = pd_index["Region"]
    new_pd = new_pd.sort_values('index')
    new_pd = new_pd[new_pd["Country"] != "ROW"]
    regions = new_pd["Region"].unique()

    horizontal_lines= []
    for region in regions:
        horizontal_lines.append(new_pd[new_pd["Region"] == region].values[-1][-2])

    sns.heatmap(new_pd[["PCA", "LPA", "Louvain", "Fuzzy"]], cmap="YlGnBu", linewidths=1)
    x1 = [0, 1, 2, 3]
    squad = [' PCA', '  LPA', ' Louvain', ' Fuzzy']


    ax1.set_xticklabels(squad,  ha ='center', minor=False)
    ax1.set_xticks(x1)
    plt.xlabel('Models')

    ax1.set_yticks(y1)
    ax1.set_yticklabels(new_pd["Country"], minor=False, rotation='horizontal')
    plt.ylabel('Country')

    for line in horizontal_lines:
        ax1.axhline(line, 0, 1, linewidth=3, c='r')
    # plt.show()
    print ax1.get_xticklabels()
    plt.savefig('/home/striker/Documents/Factor-Analysis/Graph/Cluster_2014.png')
    plt.close()

if __name__ == '__main__':
    create_new_heatmap(15)