**Predictive Analysis Using Social Profile in Online P2P Lending Market**

- The Dataset I am using is Prosper Loan Dataset it consists of 81517 Rows and 81 Columns. 
- It contains details about all the customers of the prosper company and also details of the customer whether the Loan is Accepted or Rejected.  

**Aim of the Project:** 

From the given dataset we have to predict whether the Borrower is Risk Free or is in High Risk. To predict this status, we have to build a machine learning model which will take the input values and predicts whether we accept or reject a borrower. 

**Steps involved to build the model:**   

1. Data Cleaning 
1. Exploratory Data Analysis. 
1. Feature Engineering. 
1. Model Building. 
1. Model Deployment with Streamlit and Heroku. 

**Data Cleaning:**

- Given data contains a lot of, few of them contains very less numbers of missing data which are treatable. 
- But few contains a huge number of missing data which cannot be fixed.  
- To fill the missing values in numerical columns I used median value. I categorical columns like Borrowerstate , Employmentstatus and Occupation I filled with Not available. In rest of the categorical columns filled with most frequent value. 
- Dropped those columns where more than 50% of data is missing. Reason for doing this, is simple that there is no need to keep the data from which we unable to get the sufficient information. 
- There were few outliers also in the third quantile, which I treated through Inter Quartile Range (IQR) method.  
- There was no duplicity in the dataset. 

**Exploratory Data Analysis:** 

- Since we have 81 features in our Dataset, we don't need all features for our Analysis. While performing EDA we can find the important features and we can use those selected feature for our Model Building.  
- In Our Dataset There are 3 column of Bool Data type, 50 columns of Float Data type, 11 columns of Integer Data type and 17 columns of Object Datatype. 
- Basically, we opt both the ways to analyze the dataset Univariant and Bivariant Analysis. In Univariant Analysis we studied each and every variable separately.  
- In Bivariant Analysis, we studied two variables at a time. Independent variable to independent variable or independent variable to dependent variable.  

**Analytical Insights:** 

- While plotting bar graph of "IsBorrowerHomeOwner" found that maximum people who are in rejected category, don't have their own home.  
- From the plots I got an Insight that customers who are employed paid their loan when compared to rest of the Borrowers. 
- There is a direct proportion between the IncomeRange of the borrowers and their chances of loan acceptance. As the salary increases, the chance of getting loan increases. 
- Borrowers not disclosing their income have less chances of loan approval than those who are unemployed. 
- With the Help of Boxplot, we found that the Dataset contains outliers. 
- Also, plotted a Heatmap which shows the correlation of the Variables. From heatmap, we get the list of Variables that are highly correlated with other independent variables. After selecting those variables, we further proceed to Feature Engineering part. 

**Feature Engineering:** 

Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning.  

In simple terms, we converted raw observations into desired features using statistical or machine learning approaches. 

Here we had the target variable i.e., LoanStatus consists of 11 values (Current, Completed, FinalPaymentInProgress, Chargedoff, Defaulted, Past Due (1-15 days), Past Due (16-30 days), Past Due (31-60 days), Past Due (61-90 days), Past Due (91-120 days), Cancelled) So, we created a new variable and clubbed the values 'Current', 'Completed', 'FinalPaymentInProgress' as Accepted and 'Chargedoff', 'Defaulted', 'Past Due (1-15 days)', 'Past Due (31-60 days)','Past Due (61-90 days)', 'Past Due (91-120 days)', 'Past Due (16-30 days)','Past Due (>120 days)', 'Cancelled' as Rejected. Later on, converted these two Loan Status to No Risk and High Risk. 

**Model Building:** 

- In order to build a model, we worked on three machine learning algorithms i.e., Regularized Logistic Regression, K-Nearest Neighbors (KNN) & Random Forest by using Python library sklearn.  
- Out of these three models KNN was the best performed model.  
- So, I chose KNN model for the Final Deployment. 

**KNN Model:** 

- Before proceeding for model building part, we converted all the categorical data into numerical form by using Label Encoder.   
- Created two new variables, one contains Independent Variables and another Dependent Variables/Target Variable. 
- Spitted the whole data into two parts for training, validation and testing purposes. 
- Applied KNN algorithm on the dataset. Although, KNN model is the most time taking algorithm in comparison of other two models. But it was giving an outstanding result i.e., 91% of Training Accuracy, 89% of Testing Accuracy and 84% of ROC\_AUC\_score.  
- With the help of confusion matrix, we understood how good our model at prediction. 

**Model Deployment with Streamlit and Heroku:** 

- For Model Deployment we needed an app.py file, pickle file, csv file and a model file. 
- So, we created new folder which contains all the above required files. 
- To create a Pickle file, I have imported a pickle module and with this code "pickle.dump(classifier,open("model.pkl","wb")". By this way a new pickle file is created in my folder. 
- To create the Prediction application, we used Streamlit first which allows us to have a look on our local machine.  
- Later on, we uploaded all the files on GitHub and the deployed the model by the help of Heroku, which automatically takes all the required files to deploy and create an application. 

**APP Link - [` `*Click Here* ](https://approvie.herokuapp.com/)**
