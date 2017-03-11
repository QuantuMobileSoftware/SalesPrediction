import argparse
import warnings
import pandas as pd

from time import time
from sklearn.externals import joblib


# ignore version warnings
def warn(*args, **kwargs):
    warnings.warn = warn
    pass


# prepare data to classifier
def processing(dataset):
    warn()
    dataset = dataset[features]
    dataset = dataset.drop_duplicates()
    cleaned_data = dataset
    dataset[number_features] = dataset[number_features].fillna(0)
    dataset[text_features] = dataset[text_features].fillna('NULL')
    dataset['created_at'] = pd.to_datetime(dataset['created_at'])
    dataset['created_at'] = pd.to_datetime('today') - dataset['created_at']
    dataset['created_at'] = dataset['created_at'].astype('timedelta64[D]')
    data = dataset.iloc[:, :23]
    data = pd.get_dummies(data)

    return data, cleaned_data


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
            'created_at'
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

# add argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', help='path to the .csv file')
args = vars(ap.parse_args())

# start timer
start_time = time()
print ('Start prediction')

# load .csv and .pkl files
dataset = pd.read_csv(args['file'], delimiter=',', low_memory=False)
model = joblib.load("model.pkl")

# call processing function for data
data, cleaned_data = processing(dataset)

# making prediction
predict = model.predict(data)

predict = pd.DataFrame(predict, columns=['predict'])
cleaned_data = pd.DataFrame(cleaned_data)

# sva to .csv file
result = pd.concat([cleaned_data, predict], ignore_index=True, axis=1)
result.to_csv('predicted_{}'.format(args['file']))

# end timer
end_time = time() - start_time
print ('Prediction ended in {} seconds'.format(round(end_time, 2)))
