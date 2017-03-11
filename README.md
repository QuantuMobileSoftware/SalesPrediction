# Sales Prediction

## Description
This project was created to predict purchases in the online store and  it consist two main file `Classifier.py` and `model.pkl`. The first one is Python script that prepare data sets t=in format that is good for machine learning algorithms. The second one is pre-trained machine learning model that was trained on data sets with absolutely same structure.

## Data Praparation
Here is all steps for data preparation:

1. Feature selection

  * sauce - store name
  * age_10pct - probability of cunsomer's age being in this range
  * age_25pct - probability of cunsomer's age being in this range
  * age_33pct - probability of cunsomer's age being in this range
  * age_50pct - probability of cunsomer's age being in this range
  * age_67pct - probability of cunsomer's age being in this range
  * age_75pct - probability of cunsomer's age being in this range
  * age_90pct - probability of cunsomer's age being in this range
  * gender_male_prob - probability of consumer being male based on name
  * gender_female_prob - probability of consumer being female based on name
  * no_first_name_data - consumer name unknown
  * agi_grp1_prob - probability of consumer's adjusted gross income in this range
  * agi_grp2_prob - probability of consumer's adjusted gross income in this range
  * agi_grp3_prob - probability of consumer's adjusted gross income in this range
  * agi_grp5_prob - probability of consumer's adjusted gross income in this range
  * agi_grp6_prob - probability of consumer's adjusted gross income in this range
  * no_income_data - income data for this individual is not available
  * us_state - US state
  * us_region - US region
  * ngeo - no account location identifier from geographic data
  * correct_first_name - first name info corrected
  * created_at - account creation date
  * ccreate - purchase date

2. Drop duplicates

3. Fill NaN

4. Convert data into days

5. Convert text features into vector of integers

## Machine Learning Algorithm
For this task was used Desicion Tree from Scikit-Learn that return 71,44% of accuracy and work well and Stochastic Gradient Descent that return 81,12% of accuracy but the results isn't correct, because all data was labeled with no purchases. All another algorithm give us lower results for accuracy.

## Technology
* Python
* Pandas
* Scikit-Learn

## Usage
To install all requirements and run `Classifier.py` type the following command
```
pip install -r requirements.txt
```

```
python Classifier.py -f filename.csv
```
