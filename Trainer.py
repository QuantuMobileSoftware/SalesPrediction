import argparse
import pandas as pd

from time import time
from sklearn import tree
from sklearn.externals import joblib

def is_not_null(x):
    if pd.isnull(x):
        return 0
    else:
        return 1

# prepare data to classifier
def processing(dataset):
    dataset = dataset[features]
    dataset = dataset.drop_duplicates()
    dataset[number_features] = dataset[number_features].fillna(0)
    dataset[text_features] = dataset[text_features].fillna('NULL')
    dataset['ccreate'] = dataset['ccreate'].apply(is_not_null)
    dataset['created_at'] = pd.to_datetime(dataset['created_at'])
    dataset['created_at'] = pd.to_datetime('today') - dataset['created_at']
    dataset['created_at'] = dataset['created_at'].astype('timedelta64[D]')
    data = dataset.iloc[:, :23]
    data = pd.get_dummies(data)
    label = dataset['ccreate']
    return data, label

# feature selection
features = ['sauce',
            'age_10pct',
            'age_25pct',
            'age_33pct',
            'age_50pct',
            'age_67pct',
            'age_75pct',
            'age_90pct',
            'gender_male_prob',
            'gender_female_prob',
            'no_first_name_data',
            'agi_grp1_prob',
            'agi_grp2_prob',
            'agi_grp3_prob',
            'agi_grp4_prob',
            'agi_grp5_prob',
            'agi_grp6_prob',
            'no_income_data',
            'us_state',
            'us_region',
            'ngeo',
            'correct_first_name',
            'created_at',
            'ccreate'
            ]

number_features = ['age_10pct',
                   'age_25pct',
                   'age_33pct',
                   'age_50pct',
                   'age_67pct',
                   'age_75pct',
                   'age_90pct',
                   'gender_male_prob',
                   'gender_female_prob',
                   'no_first_name_data',
                   'agi_grp1_prob',
                   'agi_grp2_prob',
                   'agi_grp3_prob',
                   'agi_grp4_prob',
                   'agi_grp5_prob',
                   'agi_grp6_prob',
                   'no_income_data',
                   'ngeo',
                   'correct_first_name'
                   ]

text_features = ['sauce',
                 'us_state',
                 'us_region'
                 ]

date_features = ['created_at',
                 'ccreate'
                 ]

# add argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', help='path to the .csv file')
args = vars(ap.parse_args())

# start timer
start_time = time()
print ('Start training')

# read .csv file
dataset = pd.read_csv(args['file'], delimiter=',', low_memory=False)

data, label = processing(dataset)

# train classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, label)

# save model as .pkl
joblib.dump(clf, "model.pkl", compress=3)

# end timer
end_time = time() - start_time
print ('Training ended in {} seconds'.format(round(end_time, 2)))
