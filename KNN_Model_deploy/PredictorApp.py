#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# In[2]:


st.write("""
# Loan Approver
The application predicts whether the borrower is eligible for the **Loan** or not.

""")


# In[3]:


st.sidebar.header('User Input Features')


# In[4]:


#Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        IncomeRange = st.sidebar.selectbox('Income Range',('$25,000-49,999','$50,000-74,999','$100,000+',
                                                           '$75,000-99,999','Not disclosed', '$1-24,999', 'Not Employed'))
        EmploymentStatus = st.sidebar.selectbox('Employment Status',('Employed','Full-time','Self-employed','Part-time',
                                                                     'Retired', 'Not employed','Other'))
        Occupation = st.sidebar.selectbox('Occupation',('Others', 'Professional', 'Engineers', 'Sales', 'Labours', 'Banking Professionals',
                                                        'Teaching Professionals','Medical Professionals', 'Executive', 'Government Jobs',
                                                        'Administrative Assistant', 'Drivers', 'Tradesman', 'Civil Service',
                                                        'Food Service Management', 'Students', 'Postal Service', 'Pharmacist', 
                                                        'Aviation Professionals', 'Religious', 'Investor','Homemaker'))
        BorrowerState = st.sidebar.selectbox('Borrower State', ("CA","TX","NY","FL","IL","GA","OH","MI","VA","NJ","NC","WA",
                                                               "PA","MD","MO","MN","MA","CO","IN","AZ","WI","OR","TN","AL",
                                                                "CT","SC","NV","KS","KY","OK","LA","UT","AR","MS ","NE",
                                                               "ID","NH","NM","RI","HI","WV","DC","MT","DE","VT","AK","SD",
                                                               "IA","WY","ME","ND"))
        TotalInquiries = st.sidebar.slider('Total Inquiries', 1, 30, 0)
        BorrowerAPR = st.sidebar.slider('Borrower Rate', 0.074190, 0.374530, 0.0)
        DebtToIncomeRatio = st.sidebar.slider('DebtToIncome Ratio', 0, 10, 0)
        Investors = st.sidebar.slider('Investors', 1, 1189, 1)
        Term = st.sidebar.slider('Term', 12,60,36)
        DelinquenciesLast7Years = st.sidebar.slider('Delinquencies Last 7Years', 0,99,0)
        MonthlyLoanPayment = st.sidebar.slider('Monthly Loan Payment', 0.0, 2251.5, 291.0)
        TotalTrades = st.sidebar.slider('Total Trades', 1,122,24)
        StatedMonthlyIncome = st.sidebar.slider('Stated Monthly Income', 0,15000,6000)
        IsBorrowerHomeowner = st.sidebar.selectbox('Is Borrower Homeowner', ('Yes', 'No'))
        CurrentlyInGroup = st.sidebar.selectbox ('Currently In Group', ('Yes', 'No'))
        AmountDelinquent = st.sidebar.slider ('Amount Delinquent', 0 , 22034, 0)
        LP_CustomerPrincipalPayments = st.sidebar.slider ('LP_Customer Principal Payments', 0.0, 20000.00, 0.0)
        LoanCurrentDaysDelinquent = st.sidebar.slider('LoanCurrent Days Delinquent', 0.0, 2190.00, 0.0)
        PublicRecordsLast12Months = st.sidebar.slider ('Public Records Last 12Months', 0, 1)
        RevolvingCreditBalance = st.sidebar.slider('Revolving Credit Balance', 0.0, 146125.5600, 0.0)
        EstimatedEffectiveYield =  st.sidebar.slider('Estimated Effective Yield', 0.052400, 0.295700, 0.0)
        AvailableBankcardCredit = st.sidebar.slider('Available Bank Card Credit', 0.0, 86737.52000, 0.0)
        LoanOriginalAmount = st.sidebar.slider ('Loan Original Amount', 1000.00, 25000.00, 0.00)
        
        data = {'IncomeRange': IncomeRange,
                'BorrowerAPR': BorrowerAPR,
                'DebtToIncomeRatio': DebtToIncomeRatio,
                'Investors': Investors,
                'Term': Term,
                'MonthlyLoanPayment': MonthlyLoanPayment,
                'EmploymentStatus': EmploymentStatus,
                'AvailableBankcardCredit': AvailableBankcardCredit,
                'TotalInquiries': TotalInquiries,
                'StatedMonthlyIncome': StatedMonthlyIncome,
                'AmountDelinquent' :AmountDelinquent,
                'DelinquenciesLast7Years': DelinquenciesLast7Years, 
                'CurrentlyInGroup': CurrentlyInGroup,
                'TotalTrades' :TotalTrades, 
                'IsBorrowerHomeowner' : IsBorrowerHomeowner,
                'LoanCurrentDaysDelinquent': LoanCurrentDaysDelinquent,
                'PublicRecordsLast12Months' : PublicRecordsLast12Months,
                'RevolvingCreditBalance' : RevolvingCreditBalance,
                'PublicRecordsLast12Months' : PublicRecordsLast12Months,
                'EstimatedEffectiveYield' : EstimatedEffectiveYield, 
                'LP_CustomerPrincipalPayments' : LP_CustomerPrincipalPayments, 
                'LoanOriginalAmount' : LoanOriginalAmount}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# In[5]:


# This will be useful for the encoding phase
Raw_data = pd.read_csv('updated_prosper_data.csv')
Selected_data = Raw_data [['Term', 'BorrowerAPR', 'TotalInquiries', 'LP_CustomerPrincipalPayments',
                           'AmountDelinquent', 'DelinquenciesLast7Years',
                           'LoanCurrentDaysDelinquent', 'PublicRecordsLast12Months',
                           'RevolvingCreditBalance', 'LoanOriginalAmount',
                           'EstimatedEffectiveYield', 'AvailableBankcardCredit', 'TotalTrades',
                           'DebtToIncomeRatio', 'StatedMonthlyIncome', 'MonthlyLoanPayment',
                           'Investors', 'Occupation', 'BorrowerState', 'EmploymentStatus',
                           'IsBorrowerHomeowner', 'CurrentlyInGroup', 'IncomeRange',
                           'LoanStatus_new']]
df_new = Selected_data.drop(columns=['LoanStatus_new'], axis=1)

df = pd.concat([input_df, df_new],axis=0)


# In[6]:


col = df.dtypes[df.dtypes == 'object'].index


# In[7]:


col


# In[8]:


# Label Encoding the variables
LE = LabelEncoder()
for i in col:
    df[i] = LE.fit_transform(df[i])
df = df[:1]  # Selects only the first row (the user input data)


# In[9]:


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')


# In[10]:


df


# In[12]:


# Reads in saved classification model
KNN = pickle.load(open('KNN_model.pkl', 'rb'))


# In[13]:


prediction = KNN.predict(df)
prediction_proba = KNN.predict_proba(df)


# In[14]:


safe_html="""  
      <div style="background-color:#8CE57A;padding:10px >
       <h2 style="color:white;text-align:center;"> No Risk</h2>
       </div>
    """
danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> High Risk</h2>
       </div>
    """
st.subheader('Prediction')
if prediction:
    st.markdown(safe_html, unsafe_allow_html=True)
else:
    st.markdown(danger_html, unsafe_allow_html=True)

st.subheader('Prediction Probability')
st.write('0 = High Risk | 1 = No Risk')
st.write(prediction_proba)


# In[ ]:





# In[ ]:




