# Abalone-Species-Data-Analysis

## Data 
This data was obtained by the UCI Machine Learning Repository. This dataset has information regarding the physical characteristics of abalone such as the sex, height, length, diameter, whole weight, shucked weight, viscera weight, shell weight, rings. 

The terms whole weight refers to the complete weight of the abalone, shucked weight is the weight of meat, viscera weight is the gut weight, shell weight is the weight after the shell has been dried, rings act as a proxy for age and it represents the number of rings counted on abalone’s shell.

The Abalone dataset underwent preprocessing to ensure data integrity, including checking for missing values and maintaining consistent data types. 
dataset source - https://archive.ics.uci.edu/ml/datasets/abalone


## Project Overview:

This project focuses on the analysis of the Abalone dataset, which contains physical measurements of marine snails. The main goal is to investigate and predict attributes such as age and sex using both statistical and machine learning approaches. The study explores crucial research questions using techniques like regression analysis, decision trees, clustering, and principal component analysis (PCA).

## Research Questions:

How accurately can the age of an abalone be predicted using its physical measurements?
What is the relationship between age and sex?
How do gender and other physical characteristics of abalones relate to each other?
Can distinct groups of abalones be identified based on shared characteristics?

## Tools and Technologies:
Programming Languages: Python, R
Data Manipulation and Analysis: pandas, numpy
Data Visualization: matplotlib, seaborn, Tableau
Machine Learning: sklearn, clustMixType
Statistical Analysis: scipy, statsmodels
Dimensionality Reduction: PCA

## Analysis and Techniques:
The project utilized various statistical and machine learning methods, including:
Linear Regression: Used to forecast age based on physical measurements.
Polynomial Regression: Applied to capture non-linear relationships.
Principal Component Analysis (PCA): Reduced dimensionality and addressed multicollinearity.
Decision Tree and Random Forest: Utilized for classifying sex based on physical features and assessing variable importance.
Chi-Square Test: Analyzed the relationship between age and sex.
K-Prototypes Clustering: Identified distinct groups of abalones based on their physical attributes and sex.

## Outcomes and Insights:
The analysis identified significant correlations between age and sex, and successfully predicted age using physical measurements. It also effectively classified abalone sex based on physical characteristics and discovered distinct groups through clustering techniques. These insights deepen the understanding of abalone biology and have practical applications in marine biology and resource management.
