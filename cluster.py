##################################################################################################
# Author Zihan Wang Date 09/08/2020
# BMI HW3
##################################################################################################
# Data comes from https://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+Behavior+Risk#
##################################################################################################
from modules import data_process, cluster_algorithm
import matplotlib.pyplot as plt
#getting data and labels from the dataset
X = data_process.get_dataset('./resources/sobar-72.csv')
y = data_process.get_labels('./resources/sobar-72.csv')

#initialize result list from different random states for plotting
result_list = []
#applying clustering algorithm and evaluate
for i in range(50):
    #print('ramdom state initialized by int',i)
    result = cluster_algorithm.k_means(X,y,i)
    result_list.append(result)

plt.plot(range(50), result_list)
plt.xlabel('Ramdom State')
plt.ylabel('Accuracy')
plt.show()
