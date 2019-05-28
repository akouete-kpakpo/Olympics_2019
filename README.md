# Olympics_2019

Describe the project.

## Feature engineering
Link requests infos to individuals infos using joins (see Alteryx flow).
Categorical encoding of categorical variables with few factors.
Target encoding for categorical variables with many factors (>50).

## Model
LGBM with class balanced
No cross validation for fine tuning -> too time consuming
Validation method to estimate prediction error.

## Possible improvements

* Dates: calculate periods
* Text columns: tf-idf vectorizer?
* town: clusterize by region