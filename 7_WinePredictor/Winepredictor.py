import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def DecisionTreeAlgorithm(data_train, data_test, target_train, target_test):
   
    Classifier = DecisionTreeClassifier()

    Classifier.fit(data_train, target_train)

    Predictions = Classifier.predict(data_test)

    Accuracy = accuracy_score(target_test, Predictions)

    return Accuracy*100


def KNeighborsClassifierAlgorithm(data_train, data_test, target_train, target_test):

    classifier = KNeighborsClassifier(n_neighbors=3)

    classifier.fit(data_train, target_train)

    predictions = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test, predictions)

    return Accuracy*100


def RandomForestAlgorithm(data_train, data_test, target_train, target_test):
    classifier = RandomForestClassifier()

    classifier.fit(data_train, target_train)

    predictions = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test, predictions)

    return Accuracy*100


def main():

    load_data = pd.read_csv("WinePredictor.csv")

    label_encoder = preprocessing.LabelEncoder()

    load_data['Alcohol'] = label_encoder.fit_transform(load_data['Alcohol'])
    load_data['Malic acid'] = label_encoder.fit_transform(load_data['Malic acid'])
    load_data['Ash'] = label_encoder.fit_transform(load_data['Ash'])
    load_data['Alcalinity of ash'] = label_encoder.fit_transform(load_data['Alcalinity of ash'])
    load_data['Magnesium'] = label_encoder.fit_transform(load_data['Magnesium'])
    load_data['Total phenols'] = label_encoder.fit_transform(load_data['Total phenols'])
    load_data['Flavanoids'] = label_encoder.fit_transform(load_data['Flavanoids'])
    load_data['Nonflavanoid phenols'] = label_encoder.fit_transform(load_data['Nonflavanoid phenols'])
    load_data['Proanthocyanins'] = label_encoder.fit_transform(load_data['Proanthocyanins'])
    load_data['Color intensity'] = label_encoder.fit_transform(load_data['Color intensity'])
    load_data['Hue'] = label_encoder.fit_transform(load_data['Hue'])
    load_data['OD280/OD315 of diluted wines'] = label_encoder.fit_transform(load_data['OD280/OD315 of diluted wines'])
    load_data['Proline'] = label_encoder.fit_transform(load_data['Proline'])

    features = list(zip(load_data['Alcohol'], load_data['Malic acid'], load_data['Ash'], load_data['Alcalinity of ash'], load_data['Magnesium'], load_data['Total phenols'], load_data['Flavanoids'],
                    load_data['Nonflavanoid phenols'], load_data['Proanthocyanins'], load_data['Color intensity'], load_data['Hue'], load_data['OD280/OD315 of diluted wines'], load_data['Proline']))
    
    target = load_data['Class']

    feature_nm = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                  'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    print("Feture name", feature_nm)
    data_train, data_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

    X = KNeighborsClassifierAlgorithm(data_train, data_test, target_train, target_test)
    print("Accuracy using Kneighbors Algorithm is :", X, "%")
    
    Y = DecisionTreeAlgorithm(data_train, data_test, target_train, target_test)
    print("Accuracy using Decision Tree Algorithm is :", Y, "%")
    
    Z = RandomForestAlgorithm(data_train, data_test, target_train, target_test)
    print("Accuracy using Random Forest Algorithm is :", Y, "%")
  
   # Matplotlib Bars for Accuracy
    x = np.array(["Kneighbors", "DescisionTree", "RandomForest"])
    y = np.array([X, Y, Z])
    plt.bar(x, y,color ='pink',width = 0.4)
    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy")
    plt.title("Wine Prediction using various Algorithms")
    plt.show()
    plt.show()

    # Pie
    # y = np.array([X, Y])
    # plt.pie(y)
    # plt.show()


if __name__ == "__main__":
    main()
