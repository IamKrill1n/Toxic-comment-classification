from corr_calculator import corr
import os

path = 'kaggle/working'
pearson_corr_matrix = []

for i, file_name in enumerate(os.listdir(path)):
    pearson_corr_matrix.append([])
    for j, file_name2 in enumerate(os.listdir(path)):
        print(f'Comparing {file_name} with {file_name2}')
        pearson_corr_matrix[i].append(corr(os.path.join(path, file_name), os.path.join(path, file_name2))['pearson_corr'])

print(pearson_corr_matrix)