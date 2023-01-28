from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def MarvellousKNeighborsClassifier():
    # 1 Load the data
    Dataset = load_iris() 
    
    Data = Dataset.data # 150 column 4 Features
    Target = Dataset.target # 150 column 1 Labels
    
    #2:Manipulate the data #shuffle=true by default
    Data_train,Data_test,Target_train,Target_test=train_test_split(Data,Target,test_size=0.5)
   
    Classifier = KNeighborsClassifier()
    
    # 3.Build the model
    Classifier.fit(Data_train,Target_train)
    
    #4:Test the model
    Predictions = Classifier.predict(Data_test)
    
    Accuracy = accuracy_score(Target_test,Predictions)
    #5:Improve --Missing
    return Accuracy
    
    
def main():
    Ret = MarvellousKNeighborsClassifier()
    print("Accuracy of Iris dataset with KNN is ",Ret * 100,"%")

if __name__=="__main__":
    main()