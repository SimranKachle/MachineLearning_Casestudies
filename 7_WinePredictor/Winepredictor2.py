from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def WinePredictor():
    #Load dataset
    wine = datasets.load_wine()
    
    #print the features 
    print(wine.feature_names)
    
    #print the labels
    print(wine.target_names)
    
    #print top 5 records
    print(wine.data[0:5])
    
    #print the wine labels
    print(wine.target)
    
    X_train,X_test,Y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3)
    
    #create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    
    #Train the model using training sets
    knn.fit(X_train,Y_train)
    
    #Predict response for test dataset
    y_pred = knn.predict(X_test)
    
    #Model Accuracy, how often is the classifier correct?
    print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))    
    
def main():
    print("Wine predictor application using k Nearest neighbors Algorithm")
    
    WinePredictor()
    
if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    
    
    