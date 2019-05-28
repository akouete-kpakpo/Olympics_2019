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

## How to use
1. First create the folder **./preparation/exports** and then run the Alteryx flow to build train and test set enriched of individuals infos.
2. Then run the Python notebook **./Notebooks/[Test] requests infos enriched of individuals infos** after creating the folder **./results**
3. If the auto submit doesn't work go fetch the output table **./results/prediction.csv** and submit it manually on the Qscore platform