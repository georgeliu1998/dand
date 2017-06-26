#!/usr/bin/python

import sys
sys.path.append("./tools/")
import pickle
import matplotlib.pyplot
from sklearn.preprocessing import MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from pprint import pprint


#######################################
### STEP 1: SELECT FEATURES TO USE  ###
#######################################

#features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'variable_to_fixed']
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                 'bonus','restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
                 'expenses', 'loan_advances', 'other', 'director_fees', 'deferred_income',
                 'long_term_incentive']

#Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


###############################
### STEP 2: REMOVE OUTLIERS ###
###############################

# The following code finds the percentages of NaN's in the dataset
# Define the function first
def calc_nan_percent(feature):
    n = 0
    for key in data_dict:
        if data_dict[key][feature] == 'NaN':
            n += 1
    percent = '{:.0%}'.format(float(n)/len(data_dict))
    return percent

# Find out all the finance features that we need to calculate NaN percentage for
all_ftrs = data_dict['LAY KENNETH L'].keys()
email_ftrs = ['to_messages', 'shared_receipt_with_poi', 'from_messages',
              'from_this_person_to_poi', 'email_address', 'from_poi_to_this_person']
all_ftrs = [ftr for ftr in all_ftrs if ftr not in email_ftrs]

# ['salary', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus',
# 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
# 'loan_advances', 'other', 'poi', 'director_fees', 'deferred_income', 'long_term_incentive']

# Calculate the NaN percentages and store in a dict
nan_percentage = {}
for feature in all_ftrs:
    nan_percentage[feature] = calc_nan_percent(feature)

# Print the dict
print("NaN Percentages:")
pprint(nan_percentage)

# The following code identifies outliers using visualization
"""
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
"""

# Now remove the identified outlier which is the 'TOTAL' data point
data_dict.pop('TOTAL', 0)

# If a data has zero value for all features, it is not useful, so should be removed as outlier
# Identify the data points:
zero_keys = []
for key in data_dict:
    n = 0
    for ftr in all_ftrs:
        if data_dict[key][ftr] == 'NaN':
            n += 1
    if n == len(all_ftrs) - 1: # excluding the 'poi' key
        zero_keys.append(key)
print("\nData Points that Have NaN's for All Features:")
print zero_keys, '\n'  # 'LOCKHART EUGENE E'

# Now remove them
for key in zero_keys:
    data_dict.pop(key, 0)


###################################  
### STEP 3: CREATE NEW FEATURES ###
###################################    
 
# Store to my_dataset for easy export below.
my_dataset = data_dict

# Create a new feature "variable_to_fixed" which is the ratio of total variable pay
# (total stock options plus bonus) to total fixed pay (total payments minus bonus)
# The following code takes into acount various scenarios when bonus, total_pay and/or
# total_stock is zero and handles correspondingly by changing the formula
for key in my_dataset:
    bonus = my_dataset[key]['bonus']
    total_pay = my_dataset[key]['total_payments']
    total_stock = my_dataset[key]['total_stock_value']
    if bonus == 'NaN':
        bonus = 0
    if total_pay == 'NaN':
        if total_stock == 'NaN':
            my_dataset[key]['variable_to_fixed'] = bonus
        else:
            my_dataset[key]['variable_to_fixed'] = total_stock + bonus
    elif total_stock == 'NaN':
        my_dataset[key]['variable_to_fixed'] = float(bonus) / (total_pay - bonus)
    else:
        my_dataset[key]['variable_to_fixed'] = float(total_stock + bonus) / (total_pay - bonus)

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "features"
print features
print "\nlabels"
print labels

"""

##################################
### STEP 4: MAKE THE ESTIMATOR ###
##################################

# Prepare the pipeline including preprocessing, feature selection and algorithm running
# Algorithms: SVC(), LinearSVC(), KNeighborsClassifier(), RandomForestClassifier(), GaussianNB() 
pipe = make_pipeline(MinMaxScaler(),
                     SelectKBest(),
                     GaussianNB())

params = {#'pca__n_components': [2],
          'selectkbest__k': [4],
          'selectkbest__score_func': [f_classif],
          #'linearsvc__C': [0.1, 1, 10, 100],
          #'linearsvc__dual': [False],
          #'linearsvc__tol': [0.000001],
          #'kneighborsclassifier__n_neighbors': [1, 5],
          #'kneighborsclassifier__weights': ['uniform'], 
          #'kneighborsclassifier__algorithm': ['auto', 'ball_tree'],
          #'kneighborsclassifier__leaf_size': [1, 10],
          #'svc__C': [0.1, 1, 10, 100],
          #'svc__kernel': ['linear', 'rbf'],
          #'svc__gamma': [0.001, 0.0001],
          #'randomforestclassifier__n_estimators': [5, 10, 20]
          }

# Split the data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.2,
                                                                            random_state=42)

# Make an StratifiedShuffleSplit iterator for cross-validation in GridSearchCV
sss = StratifiedShuffleSplit(labels_train,
                             n_iter = 20,
                             test_size = 0.5,
                             random_state = 0)

# Make the estimator using GridSearchCV and run cross-validation
print 'GridSearching with cross-validation...'
clf = GridSearchCV(pipe,
                   param_grid = params,
                   scoring = 'f1',
                   n_jobs = 1,
                   cv = sss,
                   verbose = 1,
                   error_score = 0)


#########################################
### STEP 5: MODEL FITTING AND TESTING ###
#########################################

# Fit the model using premade estimator clf
clf.fit(features_train, labels_train)

# Calculate feature scores
scores = clf.best_estimator_.named_steps['selectkbest'].scores_
scores = [round(s, 2) for s in scores] #round to 2 decimal points

# Combine with features names and rank by score
ftr_score = zip(features_list[1:], scores)
ftr_score_sorted = sorted(ftr_score,
                          key = lambda item: item[1],
                          reverse = True) 
print "\nThe Scores for All the Features are:"
pprint(ftr_score_sorted)

# Find out the features selected by SelectKBest
ftr_index = clf.best_estimator_.named_steps['selectkbest'].get_support()
ftrs = [x for x, y in zip(features_list[1:], ftr_index) if y]
print "\nThe Selected Features Are:\n", ftrs 

# Test the model using the hold-out test data
pred = clf.predict(features_test)
print '\n', "Classification Peformance Report:"
print(classification_report(labels_test, pred))


#########################################
### STEP 6: GENERATE THE PICKLE FILES ###
#########################################

dump_classifier_and_data(clf, my_dataset, features_list)


"""
"""
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
svc = SVC(kernel="linear")

rfecv = RFECV(estimator=svc, step=1, scoring='precision')
print 'start fitting'
rfecv.fit(features, labels)
print 'fitting done'
print("Optimal number of features : %d" % rfecv.n_features_)
print rfecv.support_
features=features[:,rfecv.support_]
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

"""

