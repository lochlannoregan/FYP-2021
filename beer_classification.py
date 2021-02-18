from sklearn import tree
from sklearn.metrics import accuracy_score
import shap

training_data_X = []
training_data_y = []

testing_data_X = []
testing_data_y = []

with open("data/beer_training.txt", "r") as training_data_file:
    for line in training_data_file:
        line = line.strip("\n")
        line = line.split("\t")
        training_data_y.append(line[3])
        del line[3]
        del line[6]
        training_data_X.append(line)

with open("data/beer_test.txt", "r") as testing_data_file:
    for line in testing_data_file:
        line = line.strip("\n")
        line = line.split("\t")
        testing_data_y.append(line[3])
        del line[3]
        del line[6]
        testing_data_X.append(line)

model = tree.DecisionTreeClassifier(max_depth=2)
model.fit(training_data_X, training_data_y)


predicted_training_values = model.predict(training_data_X)

print("CART training accuracy: ", accuracy_score(training_data_y, predicted_training_values))

predicted_testing_values = model.predict(testing_data_X)

print("CART testing accuracy: ", accuracy_score(testing_data_y, predicted_testing_values))

print(testing_data_X[0])

