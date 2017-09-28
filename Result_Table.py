import pandas as pd

def find_total_modularity():
    pd_a = pd.read_csv('/home/striker/Factor-Analysis/Output/Final Results/lpa_mod_country_division.csv', header=None)
    modularity_type = pd_a.iloc[:, 1:17]
    avg_bric = 0
    avg_no_oecd = 0
    avg_oecd = 0
    avg_total = 0

    count_bric = 0
    count_oecd = 0
    count_no_oecd = 0
    for index, row in modularity_type.iterrows():
        avg = 0
        row = row.tolist()
        for ind in range(0, len(row)-1):
            avg += row[ind]
        avg /= 15

        if row[len(row)-1] == 'BRIC':
            avg_bric += avg
            count_bric += 1
        elif row[len(row)-1] == 'OECD':
            avg_oecd += avg
            count_oecd += 1
        else:
            avg_no_oecd += avg
            count_no_oecd += 1

        avg_total += avg

    print count_bric, count_oecd, count_no_oecd
    avg_bric /= count_bric
    avg_oecd /= count_oecd
    avg_no_oecd /= count_no_oecd
    avg_total /= 44

    print round(avg_bric, 5), round(avg_oecd, 5), round(avg_no_oecd, 5), round(avg_total, 5)


if __name__ == '__main__':
    find_total_modularity()
