from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def k_means(X, y):
    kmeans = KMeans(n_clusters=2, random_state=0) 
    
    kmeans.fit(X)
    labels = kmeans.labels_


    # check how many of the samples were correctly labeled
    correct_labels = sum(y == labels)/float(y.size)
    print('Accuracy score with all attr: {0:0.2f}'. format(correct_labels))
    accuracy = {}
    for column in X.columns:
        Xs = X.copy()
        Xs.drop([column],axis=1, inplace=True)
        kmeans.fit(Xs)
        labels = kmeans.labels_
        correct_labels = sum(y == labels)/float(y.size)
        accuracy[column] = correct_labels
        #print(Xs.shape)
        print('Accuracy score after dropping %s {0:0.2f}'. format(correct_labels)%column)


    # accuracy = {k: v for k, v in sorted(accuracy.items(), key=lambda item: item[1],reverse=True)}
    # for k,  v in accuracy.items():
    #     print(k,v)
    x,z = zip(*accuracy.items())
    xs = []
    attr_dict = {}
    i = 0
    for item in x:
        attr_dict[chr(ord('a')+i)]=item
        xs.append(chr(ord('a')+i))
        print(chr(ord('a')+i),item)
        i+=1
    plt.plot(xs, z)

    plt.xlabel('Attributes Removed')
    plt.ylabel('Accuracy')
    
    plt.show()
   
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