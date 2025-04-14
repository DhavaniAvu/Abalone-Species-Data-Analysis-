---
title: "STAT 515 Final Project"
author: "Dhavani Avu"
date: "11/05/2023"
output:
  html_document: default
  pdf_document: default
---


#Question-1
##How well can we predict the age of an abalone based on its physical measurements?

To solve the research question, make a prediction model, you can use regression analysis, which is a statistical method that allows you to explore the relationship between a dependent variable (in this case, the abalone's age) and one or more independent variables (such as the abalone's physical measurements).

```{r message=FALSE, warning=FALSE}
# Load the necessary libraries
library(dplyr)
library(caret)
library(MASS)
library(leaps)
library(ggplot2)
library(GGally)
library(psych)
library(car)
library(tidyr)
library(gridExtra)
library(rpart)
library(tree)
library(rpart.plot)
library(caret)
library(plotly)
library(vcd)
```

```{r message=FALSE, warning=FALSE}
# Load the dataset
abalone1 <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header = FALSE)
```

```{r message=FALSE, warning=FALSE}
# Set the column names
colnames(abalone1) <- c("Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings")
```

As mentioned in the dataset description, the age in years is calculated as Rings + 1.5. So lets add 1.5 to Rings variable in the dataset.

```{r}
abalone1$Rings <- abalone1$Rings+1.5
```

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Scatterplot matrix
ggpairs(abalone1[, c(2:9)])
```

We can see that there are some strong linear relationships between some of the variables, such as length and diameter, and whole weight and shucked weight. We can also see that the distributions of some variables are skewed, such as rings.

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Correlation matrix
cor_mat <- cor(abalone1[, -1])
corrplot::corrplot(cor_mat, method = "circle", type = "upper")

```

As of now we can remove the 'Sex' variable from the model since it is categorical and not continuous and not a phyical measurement.

```{r echo=TRUE, message=FALSE, warning=FALSE}
abalone <- abalone1[,2:9]
head(abalone)
```

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Create scatter plots for each feature
plot1<- ggplot(abalone, aes(x = Length, y = Rings)) +
  geom_point(color = "#C70039") +
  labs(x = "Length", y = "Rings") +
  ggtitle(" Length vs Rings")

plot2<-ggplot(abalone, aes(x = Diameter, y = Rings)) +
  geom_point(color = "#FFC300") +
  labs(x = "Diameter", y = "Rings") +
  ggtitle(" Diameter vs Rings")

plot3<-ggplot(abalone, aes(x = Height, y = Rings)) +
  geom_point(color = "#03C6C7") +
  labs(x = "Height", y = "Rings") +
  ggtitle("Height vs Rings")

plot4<-ggplot(abalone, aes(x = Whole_weight, y = Rings)) +
  geom_point(color = "#037D3C98") +
  labs(x = "Whole Weight", y = "Rings") +
  ggtitle(" Whole Weight vs Rings")

plot5<-ggplot(abalone, aes(x = Shucked_weight, y = Rings)) +
  geom_point(color = "#581845" ) +
  labs(x = "Shucked Weight", y = "Rings") +
  ggtitle(" Shucked Weight vs Rings")

plot6<-ggplot(abalone, aes(x = Viscera_weight, y = Rings)) +
  geom_point(color = "#82E0AA" ) +
  labs(x = "Viscera Weight", y = "Rings") +
  ggtitle("Viscera Weight vs Rings")

plot7<-ggplot(abalone, aes(x = Shell_weight, y = Rings)) +
  geom_point(color = "#7D3C98") +
  labs(x = "Shell Weight", y = "Rings") +
  ggtitle(" Shell Weight vs Rings")
  
# Combine all plots into one multi-panel plot
grid.arrange(plot1, plot2, plot3, plot4, plot5, plot6, plot7, ncol = 3)
```

If the data points form a roughly straight line, then the relationship between the two variables is linear. If the data points follow a curved pattern or there are outliers, then the relationship may be nonlinear.

```{r echo=TRUE, message=FALSE, warning=FALSE}
#  Outlier Detection and Removal
# Calculate the Mahalanobis distance for each data point
dist <- mahalanobis(abalone, colMeans(abalone), cov(abalone))

# Determine the cutoff for outlier detection
cutoff <- quantile(dist, 0.975)

# Identify the outliers
outliers <- which(dist > cutoff)

# Remove the outliers
abalone <- abalone[-outliers, ]

#Data Cleaning and Preprocessing
# Handle missing values
sum(is.na(abalone))
# There are no missing values
```


The above part of code performs outlier detection and removal on a dataset "abalone". Outliers are data points that are significantly different from the majority of the data, and can skew statistical analysis and modeling results.

First, the Mahalanobis distance is calculated for each data point in "abalone". This distance measures how far each data point is from the center of the data, in terms of the covariance and mean of the variables.

Next, a cutoff point is determined using the 97.5th percentile of the Mahalanobis distances. Any data point with a Mahalanobis distance greater than the cutoff point is considered an outlier.

The outliers are then identified using the "which" function in R, which returns the indices of the data points that meet the condition.

Finally, the outliers are removed from the original dataset using indexing. The resulting dataset with the outliers removed is stored in "abalone".

We have then removed the outliers from the dataset. Also we have checked for missing values in the cleaned training set and found none.


Lets split our data into train data (80%) and test data (20%) as shown below

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Step 1: Data splitting
set.seed(123)
trainIndex <- createDataPartition(abalone$Rings, p = 0.8, list = FALSE)
trainData <- abalone[trainIndex, ]
testData <- abalone[-trainIndex, ]
```


```{r echo=TRUE, message=FALSE, warning=FALSE}
# Fit a MLR model for the training data for the physical mesuarmeants and Discuss and Interpret the result
model1 <- lm(Rings ~ . , data = trainData)
summary(model1)

# Extract RMSE value from the model summary

# Make predictions on the test data
pred <- predict(model1, newdata = testData)

# Calculate the RMSE for the predictions
rmseM1 <- sqrt(mean((testData$Rings - pred)^2))

# Print the model summary and the calculated RMSE value

cat("RMSE:", rmseM1)

```
In this step, we have fit an MLR model to the training data using the lm function from the stats package. The response variable is the number of rings (age) and the predictors are the physical measurements of the abalone. The p-value associated with the F-statistic is less than 2.2e-16, indicating that at least one predictor is significantly related to the response variable.

The p-values associated with the coefficients test the null hypothesis that the corresponding coefficient is equal to zero. The variable with a p-value less than the significance level (usually 0.05) is considered statistically significant. Also the MSE is less.

The coefficients of this model suggest that Length and Shell_weight may not be significant predictors of the response variable Rings, as their p-values are above the significance level of 0.05. On the other hand, Height, Whole_weight, Shucked_weight, Viscera_weight, Diameter all have p-values less than 0.001, indicating that they are significant predictors of Rings.

```{r echo=TRUE, message=FALSE, warning=FALSE}

# Create a data frame with the actual and predicted values
results <- data.frame(Actual = testData$Rings, Predicted = pred)

summary(results)

```

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Train the model using k-fold cross-validation 

ctrl <- trainControl(method = "cv", number = 10)
modelCv <- train(Rings ~ .,
                   data = trainData, method = "lm", trControl = ctrl)
print(modelCv)


rmseCV1  <- modelCv$results$RMSE


```


As we can see the Actual and predicted values are almost closer , we can say that this model is significant. But we can try to improve the performance by removing insignificant variables. As we discussed, we can remove the Length variable and  shell weight variable as they are not significant and have p value is nearly 0.05 and try to fit new model. Lets try using a variable selection technique for slecting the significant variables.

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Perform variable selection using the hybrid approach
regfit.full <- regsubsets(Rings ~ . , data = trainData, nvmax = ncol(trainData), method = "seqrep")
summary <- summary(regfit.full)
summary

```

```{r echo=TRUE, message=FALSE, warning=FALSE}
ggplot(data = data.frame(summary$bic), aes(x = seq_along(summary$bic), y = summary$bic)) +
  geom_line() +
  geom_point()
```
Based on this plot, we can see that at 5 we have the least bic and so selecting 5 variables is better. For knowing which variables are best for that we use the coef() funtion and obtain as below. 

```{r echo=TRUE, message=FALSE, warning=FALSE}
coef(regfit.full ,5)
```

As we can see these 5 variables Diameter Height  Whole_weight Shucked_weight Viscera_weight are best variables. Now build a regression model using those.

```{r echo=TRUE, message=FALSE, warning=FALSE}
model2 <- lm(Rings ~ Diameter +Height + Whole_weight + Shucked_weight + Viscera_weight, data = trainData)
summary(model2)
# Make predictions on the test data
pred <- predict(model2, newdata = testData)

# Calculate the RMSE for the predictions
rmseM2 <- sqrt(mean((testData$Rings - pred)^2))

# Print the model summary and the calculated RMSE value

cat("RMSE:", rmseM2)
```

We have used variable selection technique and used in the model. Compared to the before model, we got slightly better result and also the insignificat variables are removed so model2 is better. But for confirmation about which model is better, we can use Akaike Information Criterion (AIC)

AIC is a measure of the quality of a statistical model that takes into account the trade-off between model fit and model complexity. It is used to compare different models and to select the one that has the best balance between model fit and complexity. AIC is based on the likelihood function of the model and is defined as AIC = 2k - 2ln(L), where k is the number of parameters in the model and L is the likelihood function. The lower the AIC value, the better the model is considered to be.

```{r}
# Create a data frame with the actual and predicted values
results <- data.frame(Actual = testData$Rings, Predicted = pred)
summary(results)

```


```{r echo=TRUE, message=FALSE, warning=FALSE}
# Train the model using k-fold cross-validation 

# Train the model using k-fold cross-validation 
ctrl <- trainControl(method = "cv", number = 10)
modelCv <- train(Rings ~ ., data = trainData, method = "lm", trControl = ctrl)

# Extract RMSE value from the model summary
rmseCV2 <- modelCv$results$RMSE

# Print the model summary and the extracted RMSE value
print(modelCv)


```


```{r echo=TRUE, message=FALSE, warning=FALSE}
#  Select the best model
if (AIC(model1) < AIC(model2)) {
  print(paste0("model1 AIC: ", AIC(model1),", " ,"model2 AIC: ", AIC(model2) ))
  print(paste0("model1 is best model with AIC value lower than model2"))
} else {
  print(paste0("model1 AIC: ", AIC(model1),", " ,"model2 AIC: ", AIC(model2) ))
  print(paste0("model2 is best model with AIC value lower than model1"))
}
```
From the result, we can observe that model2 is better.

Next to further improve the model, lets look at the above scatter plots that we have plotted and also the correlation plot. Based on the correlation plot, observe which variables are highly correlated. And also observe the scatter plots,in a scatter plot, if the data points form a roughly straight line, then the relationship between the two variables is linear. If the data points follow a curved pattern or there are outliers, then the relationship may be nonlinear.

Looking at the scatter plots you provided, some of the variables (such as Whole_weight and Shell_weight) show a somewhat linear relationship with Rings, while others (such as Height) show a more curved or scattered pattern. It's also worth noting that some variables (such as Diameter and Length) are highly correlated, which may cause multicollinearity issues in the regression model.

 In the below model, the interaction term between "Diameter" and "Shell_weight" and "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight" these both are included to capture their combined effect on "Rings". The quadratic term of "Height" is included to model the possible nonlinear relationship between "Height" and "Rings".


```{r echo=TRUE, message=FALSE, warning=FALSE}
model3 <- lm(Rings ~  Length + Diameter +Height + Whole_weight + Shucked_weight + Viscera_weight + Shell_weight + Height^2 + Diameter:Shell_weight + Whole_weight:Shucked_weight:Viscera_weight+Shell_weight + Diameter:Length, data = trainData)

summary(model3)

# Make predictions on the test data
pred <- predict(model3, newdata = testData)

# Calculate the RMSE for the predictions
rmseM3 <- sqrt(mean((testData$Rings - pred)^2))

# Print the model summary and the calculated RMSE value

cat("RMSEM3:", rmseM3)
```

The model3  regression model seems to have a slightly better performance as compared to the model2. The model3 has a higher R-squared value  as compared to the R-squared value of the model2, indicating that the model3 explains more variance in the response variable. The adjusted R-squared value of the model3 is also slightly higher than that of the model2 , suggesting that the model3 is a better fit. Additionally, the model3 has a lower residual standard error  than the model2, indicating that the model3's predictions are closer to the actual values. 

Make predictions  for model 3 and create a visualization of the results

```{r echo=TRUE, message=FALSE, warning=FALSE}


# Create a data frame with the actual and predicted values
results <- data.frame(Actual = testData$Rings, Predicted = pred)

summary(results)

```

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Train the model using k-fold cross-validation 

ctrl <- trainControl(method = "cv", number = 10)
modelCv <- train(Rings ~ .,
                   data = trainData, method = "lm", trControl = ctrl)
print(modelCv)

# Extract RMSE value from the model summary
rmseCV3 <- modelCv$results$RMSE


```


Now lets try to perform another regression tecnique and observe the results

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Polynomial Regression
degree <- 3 # change this to experiment with different polynomial degrees
model4 <- lm(Rings ~ poly(Length, degree) + poly(Diameter, degree) + poly(Height, degree) + poly(Whole_weight, degree) + poly(Shucked_weight, degree) + poly(Viscera_weight, degree) + poly(Shell_weight, degree), data = trainData)
summary(model4)

# Make predictios on the test data
pred <- predict(model4, newdata = testData)

# Calculate the RMSE for the predictions
rmseM4 <- sqrt(mean((testData$Rings - pred)^2))

# Print the model summary and the calculated RMSE value

cat("RMSE:", rmseM4)

```


```{r}
# Create a data frame with the actual and predicted values
results <- data.frame(Actual = testData$Rings, Predicted = pred)

summary(results)

```

This is a multiple linear regression model that includes polynomial terms for each of the independent variables. The dependent variable is "Rings," and the independent variables are "Length," "Diameter," "Height," "Whole_weight," "Shucked_weight," "Viscera_weight," and "Shell_weight."

The "poly" function in R is used to create polynomial terms of a given degree for a given variable. In this model, polynomial terms of degree 1, 2, and 3 are created for each independent variable. This means that the model includes linear, quadratic, and cubic effects of each independent variable on the dependent variable.

The model output includes the estimated coefficients for each independent variable and each degree of polynomial. The intercept term represents the predicted value of the dependent variable when all independent variables are zero. The coefficients for the polynomial terms represent the change in the predicted value of the dependent variable for a unit increase in the corresponding independent variable, raised to the power of the degree of the polynomial.

Based on the model output, the adjusted R-squared value is 0.5674, indicating that the model explains about 57% of the variance in the dependent variable. The p-value for each coefficient is less than 0.05, indicating that all of the coefficients are statistically significant at the 5% level. However, some of the coefficients for the higher-degree polynomial terms are not statistically significant at the 1% level. This suggests that some of the higher-degree polynomial terms may not be necessary in the model and could be removed to simplify the model.

To determine which higher-degree polynomial terms may not be necessary in the model, we can look at the significance of the coefficients and the corresponding p-values. In this case, we can see that some of the higher-degree polynomial terms have p-values greater than 0.05, indicating that they may not be statistically significant in predicting the response variable.

For example, the coefficients for the following terms have p-values greater than 0.05:

poly(Length, degree)2
poly(Length, degree)3
poly(Diameter, degree)1
poly(Diameter, degree)2
poly(Diameter, degree)3
poly(Height, degree)2
poly(Height, degree)3
poly(Shucked_weight, degree)3
poly(Viscera_weight, degree)2
poly(Shell_weight, degree)2

Therefore, these higher-degree polynomial terms could potentially be removed from the model to simplify it without significantly affecting its performance.

```{r}
# Train the model using k-fold cross-validation 
ctrl <- trainControl(method = "cv", number = 10)
modelCv <- train(Rings ~ ., data = trainData, method = "lm", trControl = ctrl)

# Extract RMSE value from the model summary
rmseCV4 <- modelCv$results$RMSE

# Print the model summary and the extracted RMSE value
print(modelCv)

```

Also lets check the VIF values for the model. VIF (Variance Inflation Factor) is a measure of multicollinearity in a multiple regression analysis. It measures how much the variance of the estimated regression coefficient is increased due to multi collinearity in the model.



```{r echo=TRUE, message=FALSE, warning=FALSE}
# Calculate VIF values
vif(model4)

```
To assess multicollinearity, we should focus on the GVIF^(1/(2*Df)) values, which are the square root of the GVIF values adjusted for the degrees of freedom of each predictor. These values indicate the extent to which each predictor's variance is inflated due to multicollinearity, relative to a situation in which the predictor was uncorrelated with the other predictors.

In general, a GVIF^(1/(2Df)) value below 3 is often considered acceptable, indicating that the variance of the estimated regression coefficients is not overly inflated due to multicollinearity.

So in our case, you should focus on the predictors that have GVIF^(1/(2Df)) values greater than 3:

poly(Length, degree)                5.061120
poly(Diameter, degree)              5.051128
poly(Whole_weight, degree)          8.771196
poly(Shucked_weight, degree)        4.712807
poly(Shell_weight, degree)          4.029118

Based on these results lets fit a model which have more significant variables and less Vif values


```{r echo=TRUE, message=FALSE, warning=FALSE}
# Polynomial Regression

model5 <- lm(Rings ~ poly(Length, 1) + poly(Height, 1) + poly(Whole_weight, 3) + poly(Shucked_weight, 2) + I(poly(Viscera_weight,3)[,c(1,3)]) + (poly(Shell_weight,4)), data = trainData)
summary(model5)
 # Make predictions on the test data
pred <- predict(model5, newdata = testData)

# Calculate the RMSE for the predictions
rmseM5 <- sqrt(mean((testData$Rings - pred)^2))

# Print the model summary and the calculated RMSE value

cat("RMSE:", rmseM5)
```

```{r}
# Create a data frame with the actual and predicted values
results <- data.frame(Actual = testData$Rings, Predicted = pred)

summary(results)
```


```{r echo=TRUE, message=FALSE, warning=FALSE}
# Calculate VIF values
vif(model5)
```

```{r echo=TRUE, message=FALSE, warning=FALSE}
#  Select the best model
if (AIC(model4) < AIC(model5)) {
  print(paste0("model4 AIC: ", AIC(model4),", " ,"model5 AIC: ", AIC(model5)))
  print(paste0("model4 is best model with AIC value lower than model5"))
} else {
  print(paste0("model4 AIC: ", AIC(model4),", " ,"model5 AIC: ", AIC(model5)))
  print(paste0("model5 is best model with AIC value lower than model4"))
}
```

model 4 is better than model 5. Now lets compare final result model of plonomial regressiona and liniar regression.

```{r echo=TRUE, message=FALSE, warning=FALSE}
#  Select the best model
if (AIC(model3) < AIC(model5)) {
  print(paste0("model3 AIC: ", AIC(model3),", " ,"model5 AIC: ", AIC(model5)))
  print(paste0("model3 is best model with AIC value lower than model5"))
} else {
  print(paste0("model3 AIC: ", AIC(model3),", " ,"model5 AIC: ", AIC(model5)))
  print(paste0("model5 is best model with AIC value lower than model3"))
}
```


This below visualization can help you evaluate the performance of the linear regression model. If the model is good, the points on the scatter plot should be close to the red line, indicating a strong positive correlation between the actual and predicted values. If the points are scattered far from the red line, it indicates that the model is not performing well and may require further adjustments or improvements. 

So lets plot the plot for model 3 first.


```{r echo=TRUE, message=FALSE, warning=FALSE}
testData$prediction <- predict(model3, newdata = testData)
ggplot(testData, aes(x = prediction, y = Rings)) +
 geom_point(color = "seagreen", alpha = 0.7) +
 geom_abline(color = "red") +
 ggtitle("Prediction vs. Real values")
```
In our case we can see that the points are close to the line so we can assume that this is a significant model. 

The below code calculates the residuals (difference between the predicted and actual values) for the linear regression model on the testData, and creates a plot to visualize the relationship between the predicted values and the residuals.

```{r echo=TRUE, message=FALSE, warning=FALSE}
testData$residuals <- testData$Rings - testData$prediction
ggplot(data = testData, aes(x = prediction, y = residuals)) +
 geom_pointrange(aes(ymin = 0, ymax = residuals), color = "seagreen", alpha = 0.7) +
 geom_hline(yintercept = 0, linetype = 3, color = "red") +
 ggtitle("Residuals vs. Linear model prediction")
```

The plot is a visualization of the residuals (difference between the actual and predicted values) against the linear model prediction.

The x-axis represents the predicted values of the response variable, and the y-axis represents the residuals. Each point in the plot represents a data point, and the vertical line extending from the point to the x-axis shows the magnitude of the residual for that point.

The red horizontal line represents the zero residual line, indicating the points above the line have a positive residual, and the points below the line have a negative residual.

The interpretation of the plot is to check whether the residuals are randomly scattered around the horizontal line, indicating that the linear model assumptions are met, and there is no systematic bias in the residuals. Incase if Any pattern in the residuals could indicate that the linear model is not suitable for the data.

If the residuals are evenly distributed around the horizontal line, the model is likely to be valid, but if there is a pattern or trend in the residuals, it suggests that the model is not a good fit to the data. In this case, residuals are evenly distributed around the horizontal line

now lets plot for model 5

```{r echo=TRUE, message=FALSE, warning=FALSE}
testData$prediction <- predict(model5, newdata = testData)
ggplot(testData, aes(x = prediction, y = Rings)) +
 geom_point(color = "darkblue", alpha = 0.7) +
 geom_abline(color = "red") +
 ggtitle("Prediction vs. Real values")
```

```{r echo=TRUE, message=FALSE, warning=FALSE}
testData$residuals <- testData$Rings - testData$prediction
ggplot(data = testData, aes(x = prediction, y = residuals)) +
 geom_pointrange(aes(ymin = 0, ymax = residuals), color = "darkblue", alpha = 0.7) +
 geom_hline(yintercept = 0, linetype = 3, color = "red") +
 ggtitle("Residuals vs. Linear model prediction")
```
Based on the plots of 2 models, we can say that there is no much difference for both the models. for both the models, both the plots look significant. By comparing the value of the ACI , we can say that model5 is better.

Now lets perform K fold cross validation. K-fold cross-validation is a technique for model evaluation that partitions the original dataset into K folds or subsets of approximately equal size. It then trains the model on K-1 folds and tests it on the remaining fold. This process is repeated K times, each time with a different fold held out for testing. The results are then averaged across all K folds to obtain an overall estimate of model performance.


```{r}
# Train the model using k-fold cross-validation 
ctrl <- trainControl(method = "cv", number = 10)
modelCv <- train(Rings ~ ., data = trainData, method = "lm", trControl = ctrl)

# Extract RMSE value from the model summary
rmseCV5 <- modelCv$results$RMSE

# Print the model summary and the extracted RMSE value
print(modelCv)
cat("RMSE:", rmseCV5)

```
lets gather together all the MSE values into a dafarame. 
```{r}
# Create a data frame to store the RMSE values
rmseTable <- data.frame(
  Model = c("M1", "M2", "M3", "M4", "M5"),
  Train_RMSE = c(rmseM1, rmseM2, rmseM3, rmseM4, rmseM5),
  CV_RMSE = c(rmseCV1, rmseCV2, rmseCV3, rmseCV4, rmseCV5)
)
rmseTable
```


Based on the model 5 output, the adjusted R-squared value is 0.5668, model 4 is 0.57 and RSE values are also better for model4. But model4 have more insignificant varaiables and AIC is better for model5. So assume model4 is better indicating that the model explains about 57% of the variance in the Rings Variable based on the independent variable . 

Now lets go to the start again. In the starting we have seen though the predictor variables are not too correlated with the response variable, but it is  also important to consider the correlation between the predictor variables (independent variables). If there are strong correlations (either positive or negative) between the predictor variables, it may lead to multi collinearity. 


```{r echo=TRUE, message=FALSE, warning=FALSE}
# Compute the correlation matrix
corr_matrix <- cor(abalone)
corr_matrix
```



```{r echo=TRUE, message=FALSE, warning=FALSE}
pairs.panels(abalone)
```

In the correlation matrix you provided, it is important to look for any strong correlations between the predictor variables. Variables that are strongly correlated with each other may not provide unique information to the regression model, which can lead to unstable regression coefficients. Additionally, if two predictor variables are highly correlated, it may be best to remove one of them from the model to avoid multicollinearity.

To reduce the multicollinearity, we can perform a Principal Component Analysis (PCA) to transform the correlated variables into a set of uncorrelated principal components. Then, we can use these principal components as predictors in a linear regression model.

Here's the R code to perform PCA on the abalone data and then fit a linear regression model with the principal components as predictors and Rings as the response variable:


```{r echo=TRUE, message=FALSE, warning=FALSE}
# Scale the data
abalone_scaled <- scale(abalone)

# Perform PCA
pca <- prcomp(abalone_scaled, scale=TRUE)

# Check the variance explained by each principal component
summary(pca)
```
The above code gives the pricipal components. These measures of importance can help in selecting the appropriate number of principal components to retain for analysis or modeling. In this example, one could decide to retain the first two principal components, as they explain most of the variability in the data. 
Standard deviation: It represents the amount of variation in the original data that is captured by each principal component. 
Proportion of Variance: It shows the proportion of total variance in the original data that is explained by each principal component.
Cumulative Proportion: It shows the cumulative proportion of total variance explained by each principal component. 

```{r echo=TRUE, message=FALSE, warning=FALSE}
abaloneVar = pca$sdev**2
100*cumsum(abaloneVar)/sum(abaloneVar)
```

It shows the proportion of total variance in the original data that is explained by each principal component multiplied by 100. That means these are the pernectage of accurac. For example, PC1 explains 85.86% of the total variance in the data, while PC2 explains only 8.73% of the variance.

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Plot the plot to visualize the variance explained by each principal component
plot(pca, type="l")

```
Choose the number of principal components to include based on the plot

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Choose the number of principal components to include based on the plot
num_components <- 2

# Extract the first two principal components
pc1 <- pca$x[,1]
pc2 <- pca$x[,2]

```



```{r echo=TRUE, message=FALSE, warning=FALSE}
# Fit a linear regression model with the first two principal components as predictors
model6 <- lm(Rings ~ pc1 + pc2 , data=abalone)
summary(model6)
```
The output is the result of a linear regression model with the response variable "Rings" and two predictor variables "pc1" and "pc2", which were derived from a principal component analysis (PCA). Based on the model 5 output, the adjusted R-squared value is 0.9893, indicating that the model explains about 98% of the variance in the Rings Variable based on the independent variable. 



## Question 2

## Is there any significant relation between Rings of the ablone and other physical parametrs.

Load the data and convert the "Sex" variable in the dataset to a factor using the as.factor() function.
```{r echo=TRUE, message=FALSE, warning=FALSE}
# Load the Abalone dataset
data <- abalone1
#convert into a factor varaible
data$Sex <- as.factor(data$Sex)
str(data)
```
The sample() function is used to randomly select indices for the training set. The size of the training set is set to 80% of the total data. The training and testing sets are created using the selected indices.

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Split the dataset into training and testing sets
set.seed(123)
train_indices <- sample(nrow(data), nrow(data) * 0.8)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]
```

The rpart() function is used to fit a regression tree model to the training data. The formula Rings ~ . specifies that the "Rings" variable is the response variable, and all other variables in the dataset are used as predictors. The rpart.plot() function is used to visualize the fitted regression tree model.

```{r echo=TRUE, message=FALSE, warning=FALSE}

# Fit the regression tree model
model <- rpart(Rings ~ ., data = train_data)

# Visualize the tree
rpart.plot(model)

```
The predict() function is used to make predictions on the test data using the fitted model. The predictions are stored in the predictions variable. The mean squared error (MSE) is calculated by comparing the predicted values with the actual values from the test data.

```{r}
# Make predictions on the test set
predictions <- predict(model, test_data)

# Calculate the prediction MSE before pruning
mse_before <- mean((test_data$Rings - predictions)^2)
cat("Prediction MSE before pruning:", mse_before, "\n")
```
A sequence of complexity parameter (cp) values ranging from 0 to 0.05 with a step size of 0.001 is created using seq(). For each cp value, a new regression tree model is trained using the rpart() function with the specified cp value. The predictions are made on the test data, and the MSE is calculated for each model. The results are stored in the cv_errors variable. 
A plot is created with cp values on the x-axis and cross-validation MSE values on the y-axis.

```{r}
# Perform cross-validation to determine the optimal level of tree complexity
cp_values <- seq(0, 0.05, by = 0.001)
cv_errors <- sapply(cp_values, function(cp) {
  cv_model <- rpart(Rings ~ ., data = train_data, control = rpart.control(cp = cp))
  cv_results <- predict(cv_model, test_data)
  cv_mse <- mean((test_data$Rings - cv_results)^2)
  return(cv_mse)
})
plot(cp_values, cv_errors, type = "b", xlab = "Complexity Parameter (cp)", ylab = "CV MSE")
```
The cp value with the minimum cross-validation error is determined using which.min() on cv_errors. The prune() function is used to prune the original tree with the optimal cp value, creating a pruned tree. The predict() function is used to make predictions on the test data using the pruned tree model.
The predictions are stored in the pruned_predictions variable.

```{r}
# Prune the tree using the optimal cp value
optimal_cp <- cp_values[which.min(cv_errors)]
pruned_model <- prune(model, cp = optimal_cp)

# Make predictions on the test set using the pruned tree
pruned_predictions <- predict(pruned_model, test_data)
```

The MSE is calculated by comparing the pruned predictions with the actual values from the test data.

```{r}
# Calculate the prediction MSE after pruning
mse_after <- mean((test_data$Rings - pruned_predictions)^2)
cat("Prediction MSE after pruning:", mse_after, "\n") 
```

The randomForest() function from the "randomForest" package is used to build a random forest model. The formula Rings ~ . specifies that the "Rings" variable is the response variable, and all other variables in the dataset are used as predictors.
The number of trees in the forest is set to 500, and importance measures are computed.

```{r echo=TRUE, message=FALSE, warning=FALSE}
library(randomForest)
# Build a random forest model using the training data
set.seed(123)
rf_model <- randomForest(Rings~., data=train_data, ntree=500, importance = TRUE)
```

Predictions on the test data is done using random forest model and the performance is evaluated. MSE value of random model is 4.588

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Make predictions on the test data using the random forest model
predictions_rf <- predict(rf_model, newdata = test_data)

# Evaluate the performance of the random forest model on the test data
mse_rf <- mean((test_data$Rings - predictions_rf)^2)
print(paste("MSE value for random forest model is :",mse_rf))
```
Shell_weight is the important variable, found using varImpPlot()

```{r echo=TRUE, message=FALSE, warning=FALSE}
# Calculate variable importance using the random forest model
# Plot the variable importance results

varImpPlot(rf_model)

```

# Question-3 
## Is  there is any  significant association between Age and Sex ?? Use any unsupervised or supervised techniques to ans the question 

For that lets display the frequency of each gender in the dataset

```{r}
table(abalone1$Sex)
```

```{r}
# Filter the dataset to include only "male" and "female" categories
abalone_filtered <- abalone1[abalone1$Sex %in% c("M", "F"), ]
# Perform one-way ANOVA
anova_result <- aov(Rings ~ Sex, data = abalone_filtered)
# Print the ANOVA table
print(summary(anova_result))


# Filter the dataset to include only "male" and "female" categories
abalone_filtered <- abalone1[abalone1$Sex %in% c("M", "I"), ]
# Perform one-way ANOVA
anova_result <- aov(Rings ~ Sex, data = abalone_filtered)
# Print the ANOVA table
print(summary(anova_result))


# Filter the dataset to include only "male" and "female" categories
abalone_filtered <- abalone1[abalone1$Sex %in% c("I", "F"), ]
# Perform one-way ANOVA
anova_result <- aov(Rings ~ Sex, data = abalone_filtered)
# Print the ANOVA table
print(summary(anova_result))
```

As we can see that there are more Male values in the dataset, if we perform analyis for out research question the result may be baised. So lets consider equal number of rows for all the variables.

```{r}
# Create subsets with equal number of each category of Sex
n_samples <- 1307  # Set the number of samples to use for each category
female_subset <- subset(abalone1, Sex == "F")[1:n_samples,]
male_subset <- subset(abalone1, Sex == "M")[1:n_samples,]
infant_subset <- subset(abalone1, Sex == "I")[1:n_samples,]

# Combine the subsets into a new data frame
abalone_equal <- rbind(female_subset, male_subset, infant_subset)
table(abalone_equal$Sex)
```

Now the frequency is same. Lets start analysing he realtion of Sex variable with Age(i.eRings)

```{r}
# Boxplot of rings by sex
plot1<-ggplot(abalone_equal, aes(x = Sex, y = Rings, fill=Sex)) + geom_boxplot()
ggplotly(plot1)
```

From the box plot we can see that there is no much difference between male and female as both have same median values and the values also lie in similar range but we can see that the maximum and minimum values are varying. So lets further analyse this.


Lets create a new variable AgeGroup in which we include 3 age groups Young,Middle,old based on the Rings varaible.


```{r}
#create new column 
abalone_equal$AgeGroup <- NA

#populate new column
abalone_equal$AgeGroup[abalone_equal$Rings <= 10] <- "Young"
abalone_equal$AgeGroup[abalone_equal$Rings > 10 & abalone_equal$Rings < 20] <- "Middle"
abalone_equal$AgeGroup[abalone_equal$Rings >= 20] <- "Old"

str(abalone_equal)
```
A new column is created. Now lets plot a bar plot to check the frequency of values the AgeGroup

```{r}
# Create a bar plot
ggplotly(ggplot(abalone_equal, aes(x = AgeGroup, fill = AgeGroup)) + 
  geom_bar() +
  labs(x = "AgeGroup", y = "Count"))
```

From the plot we can see that there are more number of middle aged abalone very few old aged abalone and few young abalones. Lets check the count of each of those using table function.

```{r}
# Check the number of each AgeGroup group
table(abalone_equal$AgeGroup)
```

If we want to analyse correctly that if is there is any  significant association between AgeGroup and Sex, we can perform a chi-square test. It is performed when you have two categorical variables and you want to determine if there is a significant relationship between them. In this case Age group and Sex are those two

```{r}
# Create a contingency table
contingencyTable <- table(abalone_equal$AgeGroup, abalone_equal$Sex)

# Perform the Chi-Square Test
chisq.test(contingencyTable)
```
The p-value from the Chi-Square Test is less than 2.2e-16, which is considered to be statistically significant. This indicates that there is a significant association between AgeGroup and Sex. But in this case we considered Infant also as a gender and because of that the result will be defintly better as age of the infant abalone will be definelty smaller than those of male and female. So lets remove the infant varaible for analysis and then perform Chi square test to determine if there is any significant relation between their age and gender. At fisrt lets plot a bar plot for male and female to analyse the relatiion better and then perform Chi square test. 

```{r}
#create a subset for male and female 
abalone_subset <- subset(abalone_equal, Sex %in% c("M", "F"))
```

```{r}
# Create a bar plot
# Create a stacked bar plot
ggplotly(ggplot(abalone_subset, aes(x = AgeGroup, fill = Sex)) +
           geom_bar(position = "stack") +
           labs(x = "AgeGroup", y = "Count", title = "Stacked Bar Plot by AgeGroup and Sex") +
           scale_fill_manual(values = c("#F8766D" ,"darkblue")) +
           facet_wrap(~ Sex, ncol = 2))
```


```{r}
# Create a contingency table
contingencyTable <- table(abalone_subset$AgeGroup, abalone_subset$Sex)

# Perform the Chi-Square Test
testResult <-chisq.test(contingencyTable)
chisq.test(contingencyTable)
```
The p-value from the Chi-Square Test is less than 0.05 but it is very less than the previous model, and Also the Xsquared value is better in before model. so based on that we can say that for male and female considered to be less statistically significant. To confirm this lets do the follwoing process.

Cramer's V is a measure of the strength of association between two categorical variables, with values closer to 0 indicating weaker associations, and values closer to 1 indicating stronger associations. Lets do that for deciding the relationship significance.

```{r echo=TRUE, message=FALSE, warning=FALSE}

# Calculate Cramer's V
assocstats(contingencyTable)$cramer

```

The value of Cramer's V is 0.07668083, which is relatively small.  In this case, the small value of Cramer's V suggests that there is only a weak association between AgeGroup and Sex.

Based on the all the reults we can say that there is no much significant association between AgeGroup and Sex. We can say that when we include infants, the chisquare model is better but it is obvious that infant variable age will be lesser than the other 2. So adding it will give us a baised model. But when we compare between male and female, there is there is no much significant association between AgeGroup and Sex. 




Question 4:

k-prototype clustering: Considering both numerical and categorical values

```{r}
#Read the data
aba<- abalone1
```

Convert the Sex column of the aba dataset to a factor using the as.factor function. This is done to treat the Sex column as a categorical variable for clustering purposes.
Define the cat_cols vector and assigns it the value "Sex". It specifies the categorical column(s) to be used in the clustering analysis.
Define the num_cols vector and assigns it the column names "Length", "Diameter", and "Whole_weight". It specifies the numerical column(s) to be used in the clustering analysis.

```{r warning=FALSE}
library(cluster)
# Convert sex column to factor
aba$Sex <- as.factor(aba$Sex)

# Define categorical and numerical columns
cat_cols <- c("Sex")
num_cols <- c("Length", "Diameter", "Whole_weight")
```

Normalize the numerical columns specified by num_cols in the aba dataset using the scale function. The scale function standardizes the variables by subtracting the mean and dividing by the standard deviation. The resulting normalized data is stored in the aba_norm data frame.
Combine the normalized numerical columns from aba_norm and the categorical columns specified by cat_cols from the aba dataset using the cbind function. 

```{r}
# Normalize numerical columns
aba_norm <- as.data.frame(scale(aba[, num_cols]))

# Combine normalized numerical and categorical columns
aba_comb <- cbind(aba_norm, aba[cat_cols])
```
 
Load the clustMixType package, which provides the kproto function for performing K-Prototypes clustering on mixed data types.
Perform K-Prototypes clustering on the aba_comb dataset using the kproto function from the clustMixType package. It specifies the number of clusters (k = 3), the number of random starts (nstart = 25), the lambda parameter (lambda = 0.5), and the maximum number of iterations (iter.max = 50). 

```{r include=FALSE}
library('clustMixType')
# Perform K-Prototypes clustering
set.seed(123)
kp <- kproto(aba_comb, k = 3, nstart = 25, lambda = 0.5, iter.max = 50)
```

Extract the cluster labels from the kp object and assigns them to the labels variable. Also extract the cluster centers from the kp object and assigns them to the centers variable.

The clusplot function is called to visualize the resulting clusters. It takes the normalized data (aba_norm) and the cluster labels (labels) as input. The additional parameters (color = TRUE, shade = TRUE, lines = 0, main, xlab, and ylab) are used to customize the plot's appearance, such as enabling color-coded points, shaded regions, and providing appropriate titles and labels. The resulting plot shows the clusters in a scatter plot, with each point representing an observation and the points colored according to their assigned cluster.

```{r}
# Get cluster labels and centers
labels <- kp$cluster
centers <- kp$centers

# Visualize the clusters
clusplot(aba_norm, labels, color = TRUE, shade = TRUE, lines = 0,
         main = 'Abalone dataset: K-Prototypes clustering (k = 3)',
         xlab = 'Normalized variables', ylab = 'Normalized variables')
```

