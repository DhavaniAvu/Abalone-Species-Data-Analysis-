# 🐚 Abalone Age Prediction & Statistical Modeling

## 📌 Overview
This project explores statistical modeling and machine learning techniques to predict the age of abalone using physical measurements. The dataset, obtained from the UCI Machine Learning Repository, contains biological attributes of abalone specimens, including shell measurements and weights. This project aims to investigate relationships, perform variable selection, build various predictive models (linear, polynomial, PCA-based), and evaluate model performance using RMSE, AIC, and visualization tools.

---

## 🎯 Objectives
- Predict abalone age using physical measurements through regression analysis.
- Apply variable selection techniques to improve model performance.
- Detect multicollinearity and resolve it using PCA and VIF.
- Evaluate and compare different regression models.
- Explore relationships between Age and Sex using ANOVA and Chi-square tests.
- Implement unsupervised learning using K-Prototypes clustering.

---

## 📂 Project Structure

- **Abalone-Age-Prediction/**
  - `abalone.csv` — Original dataset (UCI Repository)  
  - `STAT_FinalProject.Rmd` — Rmd script containing full analysis  
  - `README.md` — Project documentation (this file)  
  - `Visualization Plots/` — (if saved separately)


---

## 📊 Dataset Details

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/abalone)
- **Target Variable:** Rings (Age = Rings + 1.5)
- **Features:** Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight

---

## 🔍 Exploratory Data Analysis

- Scatter plots & correlation matrices to assess relationships
- Mahalanobis distance for outlier removal
- Normalization and feature scaling
- Feature distributions and skewness examination

---

## 🧼 Data Preprocessing

- Removed outliers using Mahalanobis distance
- Checked for and found no missing values
- Converted categorical variable `Sex` to factor
- Created new age groupings: **Young**, **Middle**, **Old**

---

## 🧠 Models & Techniques Used

### 🔹 Linear Regression Models

- **Model 1:** Full MLR with all features  
- **Model 2:** MLR with selected significant variables  
- **Model 3:** MLR with interaction terms and polynomial transformation  

**Evaluation:** RMSE, Adjusted R², AIC, cross-validation

---

### 🔹 Polynomial Regression

- Degree-3 polynomials on all features
- Variants explored (Models 4 & 5) by removing insignificant variables
- **Model 5** optimized for VIF and AIC

---

### 🔹 Principal Component Analysis (PCA)

- Reduced multicollinearity by using PC1 and PC2 as predictors
- **Model 6** based on PCA explained ~98% variance

---

### 📈 Model Comparison Summary (RMSE)

| Model | Train RMSE | CV RMSE |
|-------|------------|---------|
| M1    | ✅         | ✅      |
| M2    | ✅         | ✅      |
| M3    | ✅         | ✅      |
| M4    | ✅         | ✅      |
| M5    | ✅         | ✅      |

> **Final Choice:** Model 4 based on best Adjusted R², AIC, and performance tradeoff.

---

## 🧪 Statistical Tests

### ✅ Hypothesis Testing

- **ANOVA:** To evaluate impact of `Sex` on `Rings`
- **Chi-Square Tests:** To assess relationship between `Sex` and `AgeGroup`
- **Cramer's V:** Strength of association between categorical variables

#### 💡 Observations

- Including all three `Sex` categories shows strong association with age.
- Excluding `Infant` group weakens statistical significance.
- Cramer's V = 0.07 indicates weak association between `AgeGroup` and `Sex` (M/F only)

---

## 🌲 Tree-Based Modeling

### 🌳 Regression Tree (rpart)

- Pruned for optimal complexity parameter `cp`
- Compared pre- and post-pruning MSE

### 🌲 Random Forest

- Built with 500 trees
- `Shell_weight` found as the most important predictor

---

## 🔍 Clustering (Unsupervised Learning)

### 💠 K-Prototypes Clustering

- Mixed-type clustering using `Sex`, `Length`, `Diameter`, `Whole_weight`
- Used `clustMixType` package for categorical + numerical data
- Visualized 3-cluster solution using `clusplot`

---

## 📉 Visualization Highlights

- Prediction vs Actual plots with `ggplot2`
- Residuals distribution to validate assumptions
- Correlation heatmaps and scatter matrix
- Variable importance from Random Forest
- Cluster visualizations using `clusplot`

---

## 📌 Key Findings

- `Height`, `Viscera weight`, `Whole weight`, and `Shucked weight` are key predictors of age.
- Interaction and polynomial models improve performance but at complexity cost.
- PCA reduces multicollinearity and explains high variance with fewer components.
- Random Forest confirms `Shell_weight` as most important feature.
- `Sex` has weak correlation with age when excluding infants.
- **Final recommendation:** Model 4 (Polynomial Regression) for balanced performance.

---

## 🛠️ Tools & Libraries

- **Language:** R  
- **Libraries:** `ggplot2`, `caret`, `MASS`, `psych`, `car`, `rpart`, `randomForest`, `clustMixType`, `GGally`, `plotly`, `corrplot`

---

## 👩‍💻 Author

**Dhavani Avu**  
M.S. Data Analytics Engineering  
George Mason University


