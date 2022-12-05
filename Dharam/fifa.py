import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Multiclass classification algorithms:- Random Forest and KNN
# Read data from csv file 
'''
NOTE: * Following columns are removed: 
            date, home_team_total_fifa_points, away_team_total_fifa_points, home_team_score, away_team_score
      * All missing values in a dataset are replaced with 0  
'''

fifa_data = pd.read_csv('fifa_matches.csv')

# fill missing values with 0
fifa_data = fifa_data.fillna(0)

# histogram of home_team_result
# fifa_data['home_team_result'].hist(bins=5)
# plt.title('Home Team Result')
# plt.show()

# store coulmns containg categorical data
categorical_features = ['home_team', 'away_team', 'home_team_continent', 'away_team_continent', 'tournament', 'city', 'country', 'neutral_location', 'shoot_out']

# One-hot Encoding -- transform categorical data to int values (represented as binary vectors)
fifa_data = pd.get_dummies(fifa_data, columns=categorical_features)

le = preprocessing.LabelEncoder()
trasformed_label = le.fit_transform(fifa_data['home_team_result']) # draw 0, loss 1, win 2
label = np.array(trasformed_label)
features = fifa_data.drop('home_team_result', axis=1)

# feature names
features_list = list(features.columns)

# Split the fifa_data into training and testing sets
# Traning set: 80% and Test set: 20% 
X_training, X_test, y_training, y_test = train_test_split(features, label, test_size = 0.2, random_state = 42)

# Train and test Random Forest Classfier
clf = RandomForestClassifier() 
print("\nTraining Random Forest classifier...")
clf.fit(X_training, y_training)
rf_prediction = clf.predict(X_test)

# Calculate accuracy of Random Forest 
rf_accuracy = accuracy_score(y_test, rf_prediction)
print("Random Forest accuracy: " + str(rf_accuracy))
# print("Random Forest accuracy: {:.2%}".format(rf_accuracy))

# Train and test KNN
print("\nTraining KNN...")
best_accuracy = 0
best_k = 1
for k in range(1,25,2):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    #Train the model
    knn_clf.fit(X_training, y_training)
    #Predict the response for test dataset
    knn_prediction = knn_clf.predict(X_test)
    accuracy = accuracy_score(y_test, knn_prediction)
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

#print best accuracy 
print(f"Accuracy for {best_k}-nearest neighbors: {str(best_accuracy)}")
