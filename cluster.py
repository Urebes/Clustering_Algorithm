##################################################################################################
# Author Zihan Wang Date 09/08/2020
# BMI HW3
##################################################################################################
# Data comes from https://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+Behavior+Risk#
##################################################################################################
from modules import data_process, cluster_algorithm

#getting data and labels from the dataset
X = data_process.get_dataset('./resources/sobar-72.csv')
y = data_process.get_labels('./resources/sobar-72.csv')

#applying clustering algorithm and evaluate
cluster_algorithm.k_means(X,y)
