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
    st.dataframe(test_df)
#     for col in Inputs:
#         if col not in ['Gender', 'Property_Area', 'Is_Graduated']:
#             test_df[col] = test_df[col].astype('float64')
#     test_df['Is_Graduated'] = test_df['Is_Graduated'].astype('int64')
    result = final_model.predict(test_df)
    result_prop = final_model.predict_proba(test_df)[0][1] * 100
    print(result,'--------------------',result_prop)
    return result, result_prop

    
def main():
#     st.image('indian_airlines.jpg')
    Gender = st.selectbox("Gender" , ['Male', 'Female'])
    Dependents = st.selectbox("No of Dependents" , [0, 1, 2, 3])
    Self_Employed = st.checkbox("Self Employed ?")    
    Is_Married = st.checkbox("Married ?")    
    Is_Graduated = st.checkbox("Graduated ?") 
    Credit_History = st.checkbox("Positive Credit History?")
    Property_Area = st.selectbox("Property Area" , ['Semiurban', 'Urban', 'Rural'])
    Monthly_ApplicantIncome = st.slider("Applicant Income (Month)", 150, 30000)
    Monthly_CoapplicantIncome = st.slider("Co-Applicant Income (Month)", 0, 10000)
    LoanAmount = st.slider("Loan Amount", 9000, 300000)
    Loan_Amount_Term = st.selectbox("Loan Amount Term (Month)" , [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
    
    if st.button("Predict", type='primary'):
        result, result_prop = prediction(Gender, Dependents, Self_Employed, Monthly_ApplicantIncome, Monthly_CoapplicantIncome,
               LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Is_Married, Is_Graduated)
        dic = {0:'Rejected', 1:'Accepted'}
        st.markdown('## Loan Status will be: '+ str(dic[result[0]]))
        st.markdown(f'## Acceptance Probability : {round(result_prop, 2)} %')

        
if __name__ == '__main__':
    main()   
