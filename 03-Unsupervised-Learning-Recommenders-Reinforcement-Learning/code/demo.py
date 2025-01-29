class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X_test):
        predictions = []
        for i in range (len(X_test)):
            distances = [self.euclidean_distance(X_test[i], x_train) for x_train in self.X_train] #calculate the distance between the test point and all the training points
            k_indices = np.argsort(distances)[:self.k] #get the indices of the k nearest points
            k_nearest_labels = [self.y_train[i] for i in k_indices] #get the labels of the k nearest points
            most_common = np.bincount(k_nearest_labels).argmax()
            #most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common) 
        return predictions
    