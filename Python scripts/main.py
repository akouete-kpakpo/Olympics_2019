%matplotlib inline

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


# Define the competition scorer
def competition_scorer(y_true, y_pred):
    return log_loss(y_true, y_pred, sample_weight=10**y_true)


# Train sample
requests_train = pd.read_csv('data/train_requests.csv', sep=',', low_memory=False, error_bad_lines=False)

# Test sample
requests_test = pd.read_csv('data/test_requests.csv', sep=',', low_memory=False, error_bad_lines=False)


cat_cols = ['animal_presence','child_to_come', 'group_type', 'long_term_housing_request', 'requester_type', 
            'victim_of_violence', 'victim_of_violence_type']
num_cols =  ['child_situation', 'district']
cat_cols_high_freq = ['group_composition_id', 'group_id', 'housing_situation_id', 'number_of_underage', 
                      'request_backoffice_creator_id', 'social_situation_id', 'town']
date_cols = ['answer_creation_date', 'group_creation_date', 'request_creation_date']
id_cols = ['group_composition_id']
text_cols = ['group_composition_label', 'housing_situation_label']
unuseful = ['group_main_requester_id']

columns = num_cols + cat_cols + cat_cols_high_freq

X = requests_train[columns].fillna('xxx')
y = requests_train['granted_number_of_nights']

X.loc[:, cat_cols + cat_cols_high_freq] = X.loc[:, cat_cols + cat_cols_high_freq].apply(lambda col: col.astype('category'))

# split between the train and the validation samples
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=37)

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(n_jobs=-1, random_state=2019, class_weight='balanced')

lgbm.fit(X_train, y_train)

competition_scorer(y_val,lgbm.predict_proba(X_val))

lgbm.fit(X, y)

X_test = requests_test[columns]
X_test.loc[:, cat_cols + cat_cols_high_freq] = X_test.loc[:, cat_cols + cat_cols_high_freq].apply(lambda col: col.astype('category'))


y_pred = lgbm.predict_proba(X_test)

predictions = pd.concat([requests_test['request_id'], pd.DataFrame(y_pred)], axis=1)

predictions.head()

import io, math, requests

# Get your token from qscore:
# 1. Go to https://qscore.datascience-olympics.com/
# 2. Chose the competition Data Science Olympics 2019
# 3. In the left menu click 'Submissions'
# 4. Your token is in the 'Submit from your Python Notebook' tab

def submit_prediction(df, sep=',', comment='', compression='gzip', **kwargs):
    TOKEN = 'f1f124332514ab749d3c55454c005e516e0c806c6e2848fecc54d5cabe7b2402a7dfcd86d68b86d4e349294e78ef86aab724c1975a82f3c8996cc9697c13f787'
    URL='https://qscore.datascience-olympics.com/api/submissions'
    df.to_csv('temporary.dat', sep=sep, compression=compression, **kwargs)
    r = requests.post(URL, headers={'Authorization': 'Bearer {}'.format(TOKEN)},files={'datafile': open('temporary.dat', 'rb')},data={'comment':comment, 'compression': compression})
    if r.status_code == 429:
        raise Exception('Submissions are too close. Next submission is only allowed in {} seconds.'.format(int(math.ceil(int(r.headers['x-rate-limit-remaining']) / 1000.0))))
    if r.status_code != 200:
        raise Exception(r.text)

submit_prediction(predictions, sep=',', index=False, comment='my submission')