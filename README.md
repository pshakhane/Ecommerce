# churn_analysis
Analysis of customers churned and non-churned.
Summary
Churn is defined as a tendency of customers who tend to terminate the contract with e-commerce companies due to influencing factors. Customer Relation Management (CRM) has critical role to investigate the customers churn pattern and trends. Data collection has implications for the CRM database, which allow customers to purchase items online.

Introduction
Churn is defined as a tendency of customers who tend to terminate the contract with e-commerce companies due to influencing factors. Customer Relation Management (CRM) has critical role to investigate the customers churn pattern and trends. In contrast, traditionally the efficiency of customer acquisition is considered to be important for the growth of the business. Domestic e-commerce businesses have incurred high acquisition cost due to saturated market and globalization. This report is going to explore the rapid growth in e-commerce data collection and the understanding of customer purchasing behaviour, retention, and acquisition. Globalization has increased the competition in the e-commerce industry; however, customers have multiple devices in the form of making informed decisions.

Presentation and Results
Data collection has implications for the CRM database, which allow customers to purchase items online. Customers can be tracked to identify the preferred login devices, whether they are going to churn, tenure, city tier, warehouse to home, preferred payment mode, gender, hours spend on app, number of devices registered, and preferred order category. The report utilizes exploratory data analysis to understand customers’ patterns and trends. In addition, descriptive statistics, logistic regression, linear regression, Random Forest, decision tree, Principal Components Analysis, and line charts were used to compare the best performing model and make informed decisions. 

Descriptive Statistics
Table 1 demonstrates descriptive statistics of the different variables. The average churn of applicants was 0.17±0.37. The average satisfaction score was 3.07±1,38 and the average order count were 2.96±2.85. 
Table 1:  Descriptive Statistics
  
 


Correlation Matrix
The power of machine learning models could help better predict accuracy of the factors that influence the behaviour of  customers Therefore, correlation matrix was generated to determine the factors. Correlation matrix validates the association across different variables as a matrix of units. The unique unit displays the correlation of binary techniques of the unit displays the amount of association among these binary techniques. Association impacts are processed across Pearson’s correlation coefficient (r) and the values that equal or less than 0.3 are considered weak. The correlation >0.3≤0.6 is considered moderate, while >0.6 is considered strong. Therefore, the target variable is churn, and the strongest variables (order count, coupon used), (customer ID and hour spend on app) are greater than 0.6 range. The moderate variables (customer ID, number of device registered), (tenure, cash back amount), (order count, day since last order), (order count, cash back amount), (cash back amount, day since last order), (coupon used, day since last order) are greater than 0.3 range.


















Figure 1: Correlation Matrix of Features

 

Histogram
It can be seen from figure 1 that the sample mean for order count is 2.9611 slightly more than the median 2, which indicates that the data is skewed to the right with the small number of churn. Based on the standard deviation of 2.8538, I can conclude that most churn customers will be in a range of approximately 2.96±2.85. In addition, most churned customers are in the range of 1 to 3 with 25% more than 1 and 25% less than 3.





Figure 2: Histogram of order count



 

Box and whisker plot
Data skewed to the right as the distance from median to the lowest order count is longer than the distance from the median to the highest order count, and the right-hand whisker is longer than the left-hand whisker. However, the right-hand box is longer than the left-hand side.








Figure 3: Box and whisker plot of order count
 

Churned and non-churned status.
Figure 5 indicates that approximately 1000 customers have churned, while 4500 customers have non-churned. I can conclude that the cost of retention to the business is too high based on the competition to the market.









Figure 4: Distribution of orders by customers churned status


 


Logistic regression
The data was split into 80% training and 20% testing, 94% indicators of strong precision and recall for non-churned status. While 64% indicates a moderate of precision and recall for churned status. The model accuracy is 89%, that is indicates a strong performance of the model.
The sensitivity and specificity calculation performed to make decision about the best model. Sensitivity articulates the proportion of non-churned status whether they have been predicted properly. While the specificity articulates the proportion of churned status whether were correctly predicted.
The results on Table 2 indicate that sensitivity and specificity have 100% of churned and non-churned that were correctly predicted by logistic regression model. 




Table 2: Logistic Regression model
Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.97      0.94       832
           1       0.78      0.54      0.64       182

    accuracy                           0.89      1014
   macro avg       0.84      0.76      0.79      1014
weighted avg       0.88      0.89      0.88      1014

Confusion Matrix:
[[804  28]
 [ 83  99]]

Sensitivity = 804/(804+83)=0.91

Specificity =  99/(99+28)=0⋅78       

Linear regression
The MSE (4.44) indicates that the data points are dispersed widely around the mean, while MAE (1.29) indicates a moderate model accuracy. I can conclude that the model accuracy is weak, with 48% R-squared, the good performing model accuracy should be close to 1.


Table 3: Linear regression model
Mean Squared Error (MSE): 4.441518262326922
Mean Absolute Error (MAE): 1.2850981706340765
R-squared (R2): 0.4823717041902641

Random Forest
The data was split into 80% training and 20% testing, 97% indicators of strong precision and recall for non-churned status. While 83% indicates a strong of precision and recall for churned status. The model accuracy is 95%, that is indicates a strong performance of the model.
The sensitivity and specificity calculation performed to make decision about the best model. Sensitivity (0.94) articulates the proportion of non-churned status whether they have been predicted properly. While the specificity (0.97) articulates the proportion of churned status whether were correctly predicted.
The results on Table 4 indicate that sensitivity and specificity have 100% of churned and non-churned that were correctly predicted by Random Forest model. 


Table 4: Random Forest Model
Classification Report:
              precision    recall  f1-score   support

           0       0.94      1.00      0.97       832
           1       0.97      0.73      0.83       182

    accuracy                           0.95      1014
   macro avg       0.96      0.86      0.90      1014
weighted avg       0.95      0.95      0.94      1014

Confusion Matrix:
[[828   4]
 [ 49 133]]
Sensitivity = 828/(828+49)=0.94

Specificity =  133/(133+4)=0⋅97       

Decision tree
The data was split into 80% training and 20% testing, 96% indicators of strong precision and recall for non-churned status. While 82% indicates a moderate of precision and recall for churned status. The model accuracy is 94%, that is indicates a strong performance of the model.
The sensitivity and specificity calculation performed to make decision about the best model. Sensitivity (0.96) articulates the proportion of non-churned status whether they have been predicted properly. While the specificity (0.84) articulates the proportion of churned status whether were correctly predicted.
The results on Table 5 indicate that sensitivity and specificity have 100% of churned and non-churned status that were correctly predicted by Decision Tree model. 



Table 5: Decision tree model

Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       832
           1       0.84      0.79      0.82       182

    accuracy                           0.94      1014
   macro avg       0.90      0.88      0.89      1014
weighted avg       0.93      0.94      0.94      1014

Confusion Matrix:
[[805  27]
 [ 38 144]]

Sensitivity = 805/(805+38)=0.96

Specificity =  144/(144+27)=0⋅84       

Principal Components Analysis
The data was split into 80% training and 20% testing, 0.92 indicators of strong precision and recall for non-churned status. While 0.48 indicates a moderate of precision and recall for churned status. The model accuracy is 87%, that is indicates a strong performance of the model.
The sensitivity and specificity calculation performed to make decision about the best model. Sensitivity (0.87) articulates the proportion of non-churned status whether they have been predicted properly. While the specificity (0.80) articulates the proportion of churned status whether were correctly predicted.
The results on Table 6 indicate that sensitivity and specificity have 100% of churned and non-churned status that were correctly predicted by Principal Components Analysis model. 


Table 6: Principal components Analysis model

Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.98      0.92       832
           1       0.79      0.34      0.48       182

    accuracy                           0.87      1014
   macro avg       0.83      0.66      0.70      1014
weighted avg       0.86      0.87      0.84      1014

Confusion Matrix:
[[816  16]
 [120  62]]

Sensitivity = 816/(816+120)=0.87

Specificity =  62/(62+16)=0⋅80       



