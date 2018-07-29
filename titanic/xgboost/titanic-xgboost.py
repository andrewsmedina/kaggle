import pandas as pd
import numpy as np

from sklearn import tree

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

# Print the `head` of the train and test dataframes
print(train.head())
print(test.head())

train["Age"] = train["Age"].fillna(train["Age"].median())

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"] = train["Embarked"].fillna("S")

train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# my_tree_one = tree.DecisionTreeClassifier()
# my_tree_one = my_tree_one.fit(features_one, target)

# print(my_tree_one.score(features_one, target))

# Impute the missing value with the median
test.Fare[152] = test.Fare.median()
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
# test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set and print them.
# my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived.
# Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])


# Check that your data frame has 418 entries


# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label=["PassengerId"])


# Create a new array with the added features: features_two
features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))

# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train["SibSp"] + train["Parch"] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))
