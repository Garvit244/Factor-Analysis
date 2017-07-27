import pandas as pd
from sklearn.metrics import silhouette_score

rotated_score = []
for index in range(2, 14):
    filename = '/home/striker/MData/Input/' + str(index) + '.csv'
    L = pd.read_csv(filename, header= None).values

    Y = []
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

        if first_qtr == 0 and second_qtr != 0:
            first_qtr = second_qtr
        elif first_qtr == 0 and third_qtr != 0:
            first_qtr = third_qtr
        Y.append(first_qtr)
    print (round(silhouette_score(L, Y, metric='euclidean'), 3))
# print rotated_score