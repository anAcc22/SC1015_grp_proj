# SC1015 Introduction to Data Science and AI

This repository contains the notebooks used for the SC1015 mini project. In
this project we aim to understand what determines a good/bad player in Call
of Duty Modern Warfare, a popular first-person shooter (fps) game released
in 2019. We have used the [Call of Duty Players Skills](https://www.kaggle.com/datasets/aishahakami/call-of-duty-players)
dataset from Kaggle.

## Contributions

- Jia Qing - Implemented the EDA, regression, and classification pipelines.
- Zhenxi   - Suggested the use of PCA and helped to find bugs.
- Han Hua  - Assisted in brainstorming ideas.

## Problem Definition

- How might we cluster players into different "skill tiers" using metrics
like one's kill-death ratio and win rate?

- What is the best model to classify players into these "skill tiers"
using predictors like one's level and total playtime.

## Reading Order

### [Exploratory Data Analysis](https://github.com/anAcc22/SC1015_grp_proj/blob/main/eda.ipynb)

In this notebook, we explore the distributions of the different variables,
remove the outliers, and engineer several new features which we believe can
assist us later on. We also establish the target variables `winRateAlt` and
`kdRatioAlt`.

### [Regression](https://github.com/anAcc22/SC1015_grp_proj/blob/main/regression.ipynb)

This notebook attempts to predict `winRateAlt` and `kdRatioAlt` using
regression techniques like Linear Regression, K-Neighbors Regression, and
several others.

### [Principal Component Analysis](https://github.com/anAcc22/SC1015_grp_proj/blob/main/pca.ipynb)

This notebook uses [scikit-learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
to first standardize the data, conduct PCA, and then launch a 5-fold
cross-validated grid search to determine the best combination of
hyperparameters.

> Interestingly, reducing our predictors to just three dimensions using
> PCA actually dampened the performance of the classifiers.

### [Clustering](https://github.com/anAcc22/SC1015_grp_proj/blob/main/clustering.ipynb)

Using K-Means Clustering, we identified four clusters that represent different
skill tiers. We then proceed to use classification techniques such as Logistic
Regression, Support Vector Classifier, and Random Forest. The performance
metric used was the macro-averaged F1 score. We conclude the notebook using
**Permutation Feature Importance** to determine the contribution of each
feature.

### [Name Analysis](https://github.com/anAcc22/SC1015_grp_proj/blob/main/name_analysis.ipynb)

This notebook briefly explored whether aspects of one's username could assist
in predicting the skill tier. For example, do the lengths of the names and
the presence of foreign characters determine if a player is good or bad? To
no avail, we weren't able to observe any significant trends.

## Key Takeaways

- In the regression context, Random Forest was the best for predicting 
`kdRatioAlt`, while Gradient Boosting was best for predicting `winRateAlt`.

- In the classification context, Support Vector Classifier triumphed the others,
although Logistic Regression was a close second.

- The Permutation Feature Importance revealed that `level`, `gamesPlayed`,
and `hitRate` were the top three most important features. The first two suggest
that to climb the skill ladder, players simply ought to play more in order
to accumulate experience.

- The importance of `hitRate` suggests that players cannot neglect their
aim too, and they should consider using aim trainers to improve the accuracy
of their shots.

## References

- <https://scikit-learn.org/stable/supervised_learning.html>
- <https://www.kaggle.com/datasets/aishahakami/call-of-duty-players>
- <https://scikit-learn.org/stable/modules/permutation_importance.html>
- <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>
- <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>
- <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>
- <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>
