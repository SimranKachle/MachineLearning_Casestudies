from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def MarvellousSVM():
    #Load dataset
    cancer = datasets.load_breast_cancer()
    
    #print name of features
    print("Features of Cancer dataset: ",cancer.feature_names)
    
    #print label type of cancer('malignant' 'benign')
    print("Labels of cancer dataset: ",cancer.target_names)
    
    #print data(feature)shape
    print("Shape of dataset is: ",cancer.data.shape)
    
    #print cancer data features(top 5 records)
    print("First 5 records are")
    print(cancer.data[0:5])
    
    #print cancer labels(0:malignant,1:benign)
    print("Target of dataset: ",cancer.target)
    
    #split dataset into training set and test set
    X_train,X_test,Y_train,Y_test=train_test_split(cancer.data,cancer.target,test_size=0.3,random_state=109)
    #70% training and 30% test
    
    #Create svm classifier
    clf=svm.SVC(kernel='linear')#Linear kernal
    
    #Train model using training sets
    clf.fit(X_train,Y_train)
    
    #Predict response for test dataset
    y_pred=clf.predict(X_test)
    
    #Model Accuracy:how often is classifier correct?
    print("Accuracy of Model is: ",metrics.accuracy_score(Y_test,y_pred)*100)
    
def main():
    MarvellousSVM()
    
if __name__=="__main__":
    main()