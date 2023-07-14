import streamlit as st
import pandas as pd
import joblib
import pickle 
import sklearn 
import datetime
from transformer import UpdateOutliers, ReCalcNumCols, DropCols
from sklearn.impute import SimpleImputer

final_model = joblib.load("final.pkl")
Inputs = joblib.load("inputs.pkl")

def prediction(Gender, Dependents, Self_Employed, Monthly_ApplicantIncome, Monthly_CoapplicantIncome,
               LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Is_Married, Is_Graduated):

    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0,"Gender"] = Gender
    test_df.at[0,"Dependents"] = Dependents
    test_df.at[0,"Self_Employed"] = float(Self_Employed)
    test_df.at[0,"Monthly_ApplicantIncome"] = Monthly_ApplicantIncome
    test_df.at[0,"Monthly_CoapplicantIncome"] = Monthly_CoapplicantIncome
    test_df.at[0,"LoanAmount"] = float(LoanAmount)
    test_df.at[0,"Loan_Amount_Term"] = float(Loan_Amount_Term)
    test_df.at[0,"Credit_History"] = float(Credit_History)
    test_df.at[0,"Property_Area"] = Property_Area
    test_df.at[0,"Is_Married"] = float(Is_Married)
    test_df.at[0,"Is_Graduated"] = int(Is_Graduated)
    test_df.at[0,"Monthly_TotalIncome"] = Monthly_ApplicantIncome+Monthly_CoapplicantIncome
    test_df.at[0,"Monthly_LoanAmount"] = LoanAmount / Loan_Amount_Term
    test_df.at[0,"Monthly_App_LoanPercent"] = (LoanAmount / Loan_Amount_Term) / Monthly_ApplicantIncome if Monthly_ApplicantIncome > 0 else 0
    test_df.at[0,"Monthly_Coapp_LoanPercent"] = (LoanAmount / Loan_Amount_Term) / Monthly_CoapplicantIncome if Monthly_CoapplicantIncome > 0 else 0
#     st.dataframe(test_df)

    result = final_model.predict(test_df)
    result_prop = final_model.predict_proba(test_df)[0][1] * 100
    return result, result_prop

    
def main():
    st.image('LoanImg.jpg')
    dic = {'No':0, 'Yes':1}
    c_dic = {'Negative':0, 'Positive':1}
    d_dic = {'0':0, '1':1, '2':2,'3+':3}
    
    dep_cols=st.columns(2)
    with dep_cols[0]:
        Gender = st.selectbox("Gender" , ['Male', 'Female'])
    with dep_cols[1]:
        Dependents = d_dic[st.selectbox("No of Dependents" , ['0', '1', '2', '3+'])]
    
    dep_cols=st.columns(2)
    with dep_cols[0]:
        Credit_History = c_dic[st.radio("Credit History ?", ('Negative', 'Positive'))]

    with dep_cols[1]:
        Self_Employed = dic[st.radio("Self Employed ?", ('No', 'Yes'))]

        
    dep_cols=st.columns(2)
    with dep_cols[0]:
        Is_Graduated = dic[st.radio("Graduated ?", ('No', 'Yes'))]
    with dep_cols[1]:
        Is_Married = dic[st.radio("Married ?", ('No', 'Yes'))]
    
    Monthly_ApplicantIncome = st.slider("Applicant Income (Month)", 150, 50000)
    Monthly_CoapplicantIncome = st.slider("Co-Applicant Income (Month)", 0, 50000)
    LoanAmount = st.slider("Loan Amount", 1000, 1000000)
    
    dep_cols=st.columns(2)
    with dep_cols[0]:
        Property_Area = st.selectbox("Property Area" , ['Semiurban', 'Urban', 'Rural'])
    with dep_cols[1]:
        Loan_Amount_Term = st.selectbox("Loan Amount Term (Month)" , [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])

    
    if st.button("Predict", type='primary'):
        result, result_prop = prediction(Gender, Dependents, Self_Employed, Monthly_ApplicantIncome, Monthly_CoapplicantIncome,
               LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Is_Married, Is_Graduated)
        dic = {0:'Rejected', 1:'Accepted'}
        st.markdown('## Loan Status will be: '+ str(dic[result[0]]))
        st.markdown(f'## Acceptance Probability : {round(result_prop, 2)} %')

        
if __name__ == '__main__':
    main()   
