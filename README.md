# Identifying who transcribed the Avila Bible dataset

## Authors 

As part of the EDS 6342 course at UH, this project was done by 

- Mansib Mursalin
- Asha Collie
- Mani Sabarish Akula
- Jahnavi Chintala

## Dataset

Dataset Link - [Avila Bible Dataset](https://archive.ics.uci.edu/ml/datasets/Avila)

## Introduction

Our group was tasked with choosing a substantial dataset and develop predictive models using various machine learning techniques. Our chosen dataset contains features extracted from 800 images of the Avila Bible. The Avila bible is a Latin copy of the Bible produced during the twelfth century by both Italy and Spain. Each sample in the dataset contains 11 features corresponding to a group of 4 consecutive rows averaged together. Each group of rows will henceforth be called a pattern. The features of each pattern represent characteristics that were inferred from a digital scan of each page. Rubricated letters and miniatures were excluded from the scans (De Stefano, 2018). The paleographic analysis by De Stefano, et. al revealed that the Avila Bible was written by 12 different scribe monks. Our goal is to predict which scribe wrote a given pattern using the available features of the pattern. These features include intercolumnar distance, upper margin, lower margin, exploitation, row number, modular ratio, interlinear spacing, weight, peak number, modular ratio/interlinear spacing, and monk.

## Pre-processing

The dataset was downloaded directly from the University of California-Irvine’s Machine Learning Repository. The data source’s original format was ‘txt’ and was converted to csv for easier readability to a python data frame. Column names were not included in the file however, additional dataset documentation provided in the repository detailed this information and it was added during pre-processing.
From observation it was discovered that the original data source was that the set labeled for testing contained more samples than the training set. As a result, the two sets were combined to be re-split later, 70/30, by the team. We also noted that the distribution of the monk variable, containing 12 classes, was imbalanced. Some classes had thousands more instances than others and likewise one particular class accounted for less than 0.05% of the samples.

At this time, the combined set was then checked for missing values and contained none. The next step in pre-processing was to check for highly correlated variables, as these can lead to redundancy in modeling and the unnecessary inclusion of features. The variable modular ratio/interlinear spacing expectedly had a high correlation with both modular ratio and interlinear spacing. As a result, this variable was dropped from the dataset and not included in any analysis.

The documentation provided with the dataset claimed that the variables were all scaled and normalized. However, using a series of plots and graphs, many outliers were detected. The solution proposed was to replace those values that fall outside of a given quantile interval. Figure 3a and 3b displays the uncleaned dataset with noise and the final cleaned dataset, respectively.

## Models

The data was modeled by 7 different algorithms individually, using optimal hyperparameters found by doing grid search. These models are 

- Linear Classifier
- Logistic Regression
- KNN
- Support Vector Machine
- Multi-layered Perceptron
- Extreme Learning Machine
- Random Forests

## Variable Selection using KNN

After running all models and obtaining results, the next step in optimizing predictions was performing feature selection. The first selection method was to be chosen from either correlation techniques, lasso regression, or k-nearest neighbor’s algorithm. Correlation observations made in the pre-processing stage—including the use of scatter plots and heatmaps—proved that the data is not linear and would not be linearly separable. Additionally, our linear models, namely logistic regression and SGD Regressor, performed at ~50% accuracy, which is almost synonymous to random guessing. Therefore, using correlation for variable selection would be a poor choice. Our best model, random forest, is incompatible with using lasso regression in its algorithm. Since the task after preliminary modeling would be to improve upon the best model, using lasso regression would also not be a fitting selection technique. For these reasons, the first feature selection method chosen was KNN.

The proposed method was a modified best subset selection, only considering combinations of 5 variables instead of all possible combinations. Optimal hyperparameters were found during grid search and using stratified k-fold cross-validation. Grid search determined the optimal hyperparameters to be “Manhattan” for the distance metric, 1 neighbor as the nearest neighbor’s parameter, and uniform weights as the weight parameter. The feature-selection KNN model iterates through each combination of 5 variables and was evaluated based on balanced accuracy.

Random Forest and SVM were both trained with the subset of variables determined by KNN. Random Forest improved accuracy on the test set by 0.86% to 99.66% but SVM performed worse on the test set with the selected variables than without at ~75%. From deduction, it seems that the selected variables oversimplified the Support Vector Machine, an algorithm praised for its performance on high-dimensional data. Thus, this may have contributed to a high bias and reduced accuracy.

## Varibale Selection using Bi-Directional Elimination 

Bi-Directional Elimination also known as Stepwise Selection is a wrapper method of feature selection. It is a combination of forward selection and backward elimination and follows a greedy search approach by evaluating all possible combinations of features with the given machine learning algorithm. It is similar to forward selection but while adding a new feature, it checks the significance of already existing features. If it finds any feature insignificant, it removes it.

The random forest classifier with optimal hyperparameters found previously was used to evaluate the effect of bi-directional elimination based on accuracy. The five features obtained from this method happened to be the same obtained from the KNN selection technique. It was not surprising, then, that it performed with a similar accuracy of ~99.6% on the test set.

## Clustering

Clustering is a machine learning technique that involves grouping together similar data points based on a similarity metric or distance measure. The purpose of performing clustering on this data is to help identify patterns and relationships. Two clustering techniques were used on this dataset to find similarities.

K-means is a popular clustering algorithm that aims to partition a dataset into a fixed number (k) of clusters, where each data point belongs to the cluster whose center is nearest to the point. Because it is necessary to explicitly state the number of clusters (k), the elbow method was used to determine the optimal number of clusters. The elbow method involves plotting the graph of sum of squared errors vs. number of clusters and selecting k such that the increasing number of clusters will not significantly reduce the sum of squared errors. The graph plots values of k between 2 and 9 and the optimal k select was k=5.

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is another popular clustering algorithm that groups together data points based on their spatial density. The algorithm defines a radius around each data point and searches for other points within that radius. DBSCAN can be more useful for highly dimensional data because it can identify clusters of arbitrary shape and can also detect noise points that do not belong to any cluster. To determine the best value for epsilon, the radius of each data point, the k-distance graph was plot and observed for where the function’s gradient shifts toward infinity. The optimal epsilon was found to be 3 and the minimum number of points to consider a cluster was set at 10.

K-Means and DBSCAN were evaluated based on their silhouette score. The silhouette coefficient is a measure of how similar a data observation is to its own cluster compared to other clusters, and the silhouette score is the average coefficient over all data instances. The silhouette score ranges between -1 and +1 and is positively correlated with confidence in the data points’ cluster assignments. K-Means clustering had an average silhouette score of 0.215 and DBSCAN had a score of 0.401. This means that the DBSCAN has more confidence that the data points were assigned to their correct clusters. This might have been useful to build a predictive model had we returned to variable selection and considered variable combinations of 3, however the team did not have enough time to perform this.

## Ensemble

Ensemble modeling is the combining of multiple machine learning algorithms to better the performance of any individual model. They typically have better accuracy, and reduce both bias and variance. The basic idea of ensemble learning comes from the principles of ‘the wisdom of the crowd’, in that having a group of models evaluate predictions increases the likelihood that the prediction is correct.
The ensemble model created with this dataset uses all individual models built thus far with their respective optimal hyperparameter and combines them using a majority voting classifier. This means that an instance is assigned to the class that most of the models assigned it to. However, our ensemble model performed worse than the best individual model, only achieving an accuracy of ~85%. This was presumed to be because of the poor performance of the individual linear classifiers used in the ensemble, which both had accuracies of ~50%.

## Conclusion 

In conclusion, our Random Forest Classifier with feature selection performed the best with an accuracy of 99.66%. Realistically, the model does an excellent job at predicting which scribe monk wrote which pattern on new data the model has not seen. Although it is not highly likely that more time would have resulted in better results, we were eager to build the experiment and believe that we might have obtained more robust results.
