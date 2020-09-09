from sklearn.cluster import KMeans
def k_means(X, y):
    kmeans = KMeans(n_clusters=2, random_state=0) 

    kmeans.fit(X)
    labels = kmeans.labels_

    # check how many of the samples were correctly labeled
    correct_labels = sum(y == labels)/float(y.size)
    print('Accuracy score: {0:0.2f}'. format(correct_labels))