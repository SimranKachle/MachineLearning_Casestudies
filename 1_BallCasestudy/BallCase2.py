from sklearn import tree

# Load the dataset
# 1st list is list of feature and 2nd is of label

# Rough  1
# Smooth  0

# Cricket 2
# Tennis  1
def BallPredictor():
    Features = [[35, "1"], [47, "1"], [90, "0"], [48, "1"], [90, "0"], [35, "1"], [92, "0"], [35, "1"], [
        35, "1"], [35, "1"], [96, "0"], [43, "1"], [110, "0"], [35, "1"], [95, "0"]]  # list of list
    Labels = ["1", "1", "2", "1", "2", "1", "2","1", "1", "1", "2", "1", "2", "1", "2"]

    # Decide the ML Alogorithm
    obj = tree.DecisionTreeClassifier()

    # Perform the  Training  of model
    obj = obj.fit(Features, Labels)

    # Perform the testing
    print(obj.predict([[97, 0],[35,1]]))

def main():
    print("--------------------Ball Predictor Case Study--------------------")
    
    BallPredictor()    
    
if __name__=="__main__":
    main()