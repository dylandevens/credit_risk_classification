# credit_risk_classification
credit risk classification project



This challenge explored a logistic regression model to guide use of a supervised mchine learning model. A sample dataset with 8 features (1 being the target (loan_status [0(safe),1(risky)]) was used to fit a satandard LogisticRegression model. Data was split between training and test data, and a score, sonfusion matrix, and classification report were used in assessing the model's accuracy. 


Below is a list of topics covered:
* pandas
* numpy
* sklearn.metrics  confusion_matrix and classification_report
* sklearn.model_selection  train_test_split
* sklearn.linear_model LogisticRegression
* sklearn.preprocessing  StandardScaler

* pandas dataframes
* StandardScaler
* models: define, fit, predict
* model assessment: confusion matrix, classification report
* hyperparameter adjustment
* findings summary and report


todo: 
* incorporate a for loop to test different models all at once with the scaled data
*  or different parameters



***************************************************************


# Credit Risk Analysis Report

## Purpose:

The purpose of this analysis is to provide the agency with key findings to guide the decision making process of future loan applicants to mitigate risk and reduce financial loss. 

## Data

The financial data used in this analysis includes key factors of financial loan recipients, including:
* loan_size
*  amount of money loaned to borrower
* interest_rate
*  the intial interest rate for the loan
* borrower_income
*  yearly income of the barrower
* debt_to_income
*  debt to income ratio for borrower (lower is better)
* num_of_accounts
*  count of credit accounts
* derogatory_marks
*  sum of deragatory marks on credit report
* total_debt
*  sum of debt owed by borrower
* loan_status
*  assessment of borrower's risk. 0=healthy loan  1=high-risk loan

The data set contains data for 77,536 individual borrowers and does not have any personally identifiable information.

## Testing

Initial testing of the data is done with the raw, unscaled data. This data is then split into a training subgroup, and a test subgroup with the loan_status removed from the training and test data. The model uses logistic regression to train and fit to this data, and the results are saved.

Further testing is done after scaling the data's features. Then, the scaled data is split into training and testing subgroups, and used to train and fit another logistic regression model. The models' results are saved and compared to each other.

## Results

#### Model 1
* Score:
*  Training - 99.29%
*  Test - 99.25%
* Confusion Matrix:
*  True Negative - 18655
*  False Positive - 110
*  False Negative - 36
*  True Positive - 583
* Precision:
*  Healthy Loan (0) - 1.0
*  High-Risk Loan (1) - 0.84
* Recall:
*  Healthy Loan (0) - 0.99
*  High-Risk Loan (1) - 0.94


#### Model 2 (Scaled)
* Score:
*  Training - 99.43%
*  Test - 99.37%
* Confusion Matrix:
*  True Negative - 18652
*  False Positive - 113
*  False Negative - 9
*  True Positive - 610
* Precision:
*  Healthy Loan (0) - 1.0
*  High-Risk Loan (1) - 0.84
* Recall:
*  Healthy Loan (0) - 0.99
*  High-Risk Loan (1) - 0.99


## Summary

) Score
)  The scores for the training and testing data increased by about .15% for model 2 (Scaled)
)   Even though this is not an objectively high number, this increase could still be significant, and by only looking at the scores, we wouldn’t know what aspects of Model 2 were improved.

) Confusion Matrix
)  The main change here is an increase in True Positive values (and a corresponding decrease in False Negative values)  with Model 2 (Scaled).
)  This would mean that more High-Risk Loans were identified in Model 2
. The other aspects of the confusion matrix remained similar and the differences seem insignificant.

. Classification Report
.  Precision
.   Healthy Loan (0)
.    These values remained equal, with a value of 1.0 signifying that all the predicted healthy loans (0) were correct.
.   High-Risk Loan (1)
.    For both models, 84% of high-risk loans were predicted correctly.
.  Recall
.   Healthy Loan (0)
.    Out of all the healthy loans in the test dataset, 99% were predicted correctly in both models
.   High-Risk Loan (1)
.    This is where Model 2 had made improvements compared to the first model. There was a decrease from 36 to 9 False Negative values, which improved the recall from94% to 99%
.    This means that out of all the high-risk loans in the test data, 99% were correctly predicted.

. Conclusion
.  Model 2(Scaled) outperformed the initial logistic regression model. There were improvements in its accuracy score, and a significant reduction in false negatives. Scaling the data that the model trained and tested on led to marked improvements across the board.

. Recommendations 

Lending can be risky, and there is money to be made in managing and reducing risks for the lender. By studying data from borrowers and their resulting classification as a healthy loan or a high-risk loan, we have created a model to help inform future loan decisions. 
There are two aspects that are important, providing healthy loans to as many people as possible, and refusing loans to applicants who would be at a high-risk of defaulting. The first aspect leads to income, and the second aspect is responsible for financial loss. Therefore, the latter aspect is more important to a lending company, because with too many high-risk loans or loans that can lead to a significant loss, the business itself can be put at risk.
Based on these assumptions, my recommendation is as follows:

The model 2 (scaled) not only predicted  with a high degree of certainty healthy loans, but it did a very good job of reducing the amount of high-risk loans. It reduced the amount of loans that were high-risk but classified as healthy, therefore preventing 26 high-risk loans (out of 619) from being classified as healthy loans. This model does a very good job of mitigating risk, as well as correctly predicting applicants that would be favorable recipients of loans. 





***************************************************************











## Background
In this Challenge, you’ll use various techniques to train and evaluate a model based on loan risk. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## Before You Begin
* Create a new repository for this project called credit-risk-classification. Do not add this homework to an existing repository.

* Clone the new repository to your computer.

* Inside your credit-risk-classification repository, create a folder titled "Credit_Risk."

* Inside the "Credit_Risk" folder, add the credit_risk_classification.ipynb and lending_data.csv files found in the "Starter_Code.zip" file.

* Push your changes to GitHub.


## Instructions
The instructions for this Challenge are divided into the following subsections:

* Split the Data into Training and Testing Sets

* Create a Logistic Regression Model with the Original Data

* Write a Credit Risk Analysis Report

## Split the Data into Training and Testing Sets
Open the starter code notebook and use it to complete the following steps:

Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

#### NOTE
A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

Split the data into training and testing datasets by using train_test_split.

## Create a Logistic Regression Model with the Original Data
Use your knowledge of logistic regression to complete the following steps:

Fit a logistic regression model by using the training data (X_train and y_train).

Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

Evaluate the model’s performance by doing the following:

Generate a confusion matrix.

Print the classification report.

Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

## Write a Credit Risk Analysis Report
Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:

An overview of the analysis: Explain the purpose of this analysis.

The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.

A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.

## Requirements
### Split the Data into Training and Testing Sets (30 points)
To receive all points, you must:

Read the lending_data.csv data from the Resources folder into a Pandas DataFrame. (5 points)

Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns. (10 points)

Split the data into training and testing datasets by using train_test_split. (15 points)

### Create a Logistic Regression Model (30 points)
To receive all points, you must:

Fit a logistic regression model by using the training data (X_train and y_train). (10 points)

Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model. (5 points)

Evaluate the model’s performance by doing the following:

Generate a confusion matrix. (5 points)

Generate a classification report. (5 points)

Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels? (5 points)

### Write a Credit Risk Analysis Report (20 points)
To receive all points, you must:

Provide an overview that explains the purpose of this analysis. (5 points)

Using a bulleted list, describe the accuracy, precision, and recall scores of the machine learning model. (5 points)

Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning. (10 points)

### Coding Conventions and Formatting (10 points)
To receive all points, you must:

Place imports at the top of the file, just after any module comments and docstrings and before module globals and constants. (3 points)

Name functions and variables with lowercase characters, with words separated by underscores. (2 points)

Follow DRY (Don’t Repeat Yourself) principles, creating maintainable and reusable code. (3 points)

Use concise logic and creative engineering where possible. (2 points)

### Code Comments (10 points)
To receive all points, your code must:

Be well commented with concise, relevant notes that other developers can understand. (10 points)
