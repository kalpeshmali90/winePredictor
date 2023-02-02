
from sklearn import metrics
from sklearn import  datasets
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import  train_test_split
def winePredictor():
    #load the daTASET
    wine=datasets.load_wine()

    #print the name of feature
    print(wine.feature_names)

    #Print the lable species
    print(wine.target_names)

    #print the wine data(top 5 records)
    print(wine.data[0:5])

    #print the wine lables
    print(wine.target)

    #split the dataset into training and test 70% training 30% testing
    x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)

    #create KNN classifier
    knn=KNeighborsClassifier(n_neighbors=3)

    #train the model using training set
    knn.fit(x_train,y_train)

    #predict response using training data set
    y_pred=knn.predict(x_test)

    #model accuracy
    print(f"Accuracy: {metrics.accuracy_score(y_test,y_pred)}")
def main():
    print("This program will classify wines according to their classes")
    print()
    winePredictor()


if __name__=="__main__":
    main()