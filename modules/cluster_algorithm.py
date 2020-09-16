from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_means(X, y, random_state_int):
    #since the label is only 0 and 1, so 2 clusters (no need to find the maximum # of clusters)
    kmeans = KMeans(n_clusters=2, random_state=random_state_int) 
    
    kmeans.fit(X)

    #get the results
    labels = kmeans.labels_


    # check how many of the samples were correctly labeled
    correct_labels = sum(y == labels)/float(y.size)
    result = correct_labels

    #print('Accuracy score: {0:0.2f}'. format(correct_labels))

    #find out the result after removing each one of the attribute
    accuracy = {}
    for column in X.columns:
        Xs = X.copy()
        Xs.drop([column],axis=1, inplace=True)
        kmeans.fit(Xs)
        labels = kmeans.labels_
        correct_labels = sum(y == labels)/float(y.size)
        accuracy[column] = correct_labels
        #print(Xs.shape)
        #print('Accuracy score after dropping %s {0:0.2f}'. format(correct_labels)%column)

    #plot the accuracies against each removed attribute
    x,z = zip(*accuracy.items())
    xs = []
    attr_dict = {}
    i = 0
    
    #the attribute name is too long, substitute them into letters. 
    for item in x:
        attr_dict[chr(ord('a')+i)]=item
        xs.append(chr(ord('a')+i))
        #print(chr(ord('a')+i),item)
        i+=1

    # plt.plot(xs, z)
    # plt.xlabel('Attributes Removed')
    # plt.ylabel('Accuracy')
    #plt.show()
    return result
     # accuracy = {k: v for k, v in sorted(accuracy.items(), key=lambda item: item[1],reverse=True)}
    # for k,  v in accuracy.items():
    #     print(k,v)
    # Xs = X
    # for column in accuracy.keys():   
    #     Xs.drop([column],axis=1, inplace=True)
    #     kmeans.fit(Xs)
    #     labels = kmeans.labels_
    #     correct_labels = sum(y == labels)/float(y.size)
    #     accuracy[column] = correct_labels
    #     #print(Xs.shape)
    #     print('Accuracy score after dropping %s {0:0.2f}'. format(correct_labels)%column)
    # X.drop([''], axis=1, inplace=True)