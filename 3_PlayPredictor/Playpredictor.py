import numpy as py
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def PlayPredictor(path):

    # step 1 : Load data
    data = pd.read_csv(path, index_col=0)

    print("Size of Actual dataset", len(data))

    # step 2: clean,prepare and manipulate data
    feature_names = ['Whether', 'Temperature']

    print("Names of Features: ", feature_names)

    whether = data.Whether
    Temperature = data.Temperature
    play = data.Play

    #Creating labelEncoder
    le = preprocessing.LabelEncoder()
    
    #Converting string labels into numbers
    weather_encoded = le.fit_transform(whether)
    # print(whether)
    print(weather_encoded)

    temperature_encoded = le.fit_transform(Temperature)
    # print(Temperature)
    label = le.fit_transform(play)

    print(temperature_encoded)

    #Combining weather and temp into single list of tuples
    features = list(zip(weather_encoded, temperature_encoded))
    # print(features)
    
    #Step 3: Train the data
    model = KNeighborsClassifier(n_neighbors=3)

    #train model using training sets
    model.fit(features, label)

    #step 4:Test data
    predicted = model.predict([[0, 2]])

    print(predicted)

    if predicted:
        print("You can play")
    else:
        print("You cannot play")


def main():
    print("Play Predictor Application using KNN")

    PlayPredictor("PlayPredictor.csv")


if __name__ == "__main__":
    main()
