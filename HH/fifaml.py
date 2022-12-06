import csv
import random
import time
from sklearn.naive_bayes import CategoricalNB
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

'''
Method: Naive Bayes via SciKitLearn
Advantages: Can be used for multi-classification. Usually used in sentiment analysis, spam filter, text classification
Disadvantages: Assumes independence, requires supervised learning

GaussianNB: Use when feature values of samples are continuous
BernoulliNB: Use when feature values & class are binary
MultinomialNB: Counts number of outcomes of feature values over all samples.
CategoricalNB: Use when feature values are categorical, may fail if training did not show each possible feature value
ComplementNB: Like Multinomial, but for imbalanced dataset

Results:

For some reason, Bernoulli gives near equal results to Categorical, 56%.
Gaussian provides 28%.
Multinomial & Complement give about 47%

If using only home_team, away_team, fifa ranks, scores, neutral loc, shootout,
then accuracy = 58%

Decision Tree Classifier w/ & w/o extra preprocessing yields 49%
    Yields 55% when using max_depth=2, and grouping Ranks.

Decision Tree Classifier w/o extra preprocessing & w/o converting numerical feature values yields 99% ?
    The extra preprocessing dramatically reduces the accuracy below 50%
    Removing home score & away score causes this
    
SVM: Yielded 51% for categorical preprocessing & 58% without categorical preprocessing. Takes way longer to execute
    Making C=10, did not really affect accuracy. Sigmoid and poly don't perform higher than 50%
    Linear takes too long.
    
    Works best at default settings, without transforming numerical rank data to categorical. AND extra feature removal.
    
KNN: Yielded 57%. Works best at around 20 neighbors. Without binning. 4 Seconds each iteration.

RandomForest: Yielded 58% with max_depth=4 and without binning. 26 seconds each iteration
'''


def preprocess(dataset, headers, binning):
    pre_time = time.time()
    # Helper map to index & transform the categorical feature values
    val_map = []
    for col in headers:
        val_map.append({})

    # Helper array for binning. Transforms numerical data into categorical
    bins = []
    for feature in range(len(headers)):
        ranges = []
        # Find max value of a numerical feature set & create feature value ranges
        if dataset[0][feature].isnumeric():
            max_val = 0
            for sample in dataset:
                num = int(sample[feature])
                if num > max_val:
                    max_val = num
            ranges = [max_val / 5, max_val / 4, max_val / 3, max_val / 2, max_val]
        bins.append(ranges)

    # Transform all non-numerical data in the CSV
    for sample in dataset:
        for feature in range(len(headers)):
            if not sample[feature].isnumeric():
                # Update map for new items under specific header.
                if sample[feature] not in val_map[feature].keys():
                    val_map[feature][sample[feature]] = len(val_map[feature].values()) + 1

                sample[feature] = val_map[feature][sample[feature]]
            else:
                if binning:
                    # Numerical feature values become normalized into categorical numbers
                    orig_val = int(sample[feature])
                    converted_val = 0
                    for threshold in bins[feature]:
                        if orig_val > threshold:
                            converted_val += 1
                    sample[feature] = converted_val
                else:
                    sample[feature] = int(sample[feature])

    print(f'Reading & Preprocessing dataset completed in {(time.time() - pre_time)} secs')
    return dataset


def train_test(data, training_portion, model_class, iterations):
    portion = int(len(data) * training_portion)
    highest_accuracy = 0
    failures = 0
    start_time = time.time()

    for i in range(iterations):
        x_training = []
        y_training = []
        x_test = []
        y_test = []

        try:
            random.shuffle(data)
            training_set = data[:portion]
            test_set = data[portion:]

            for training_sample in training_set:
                x_training.append(training_sample[:-1])
                y_training.append(training_sample[-1])

            for test_sample in test_set:
                x_test.append(test_sample[:-1])
                y_test.append(test_sample[-1])

            model = model_class
            model.fit(x_training, y_training)
            error = 0

            for j in range(len(y_test)):
                prediction = model.predict([x_test[j]])[0]
                if prediction != y_test[j]:
                    error += 1

            accuracy = (len(y_test) - error) / len(y_test)

            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
        except IndexError:
            failures += 1

    result = f'The highest accuracy of {model_class} is {highest_accuracy}. It took {(time.time() - start_time)} secs.'
    if failures > 0:
        result += f'It failed {failures} times.'
    print(result)


dataset = []
with open('reduced_dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for feature, row in enumerate(reader):
        dataset.append(row)

headers = dataset.pop(0)
processed_dataset = preprocess(dataset, headers, False)

train_test(processed_dataset, 0.8, CategoricalNB(), 10)
train_test(processed_dataset, 0.8, tree.DecisionTreeClassifier(max_depth=10), 10)
train_test(processed_dataset, 0.8, KNeighborsClassifier(n_neighbors=2000), 1)
train_test(processed_dataset, 0.8, svm.SVC(), 1)
train_test(processed_dataset, 0.8, RandomForestClassifier(max_depth=4), 1)
