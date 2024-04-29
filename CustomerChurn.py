import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set page configuration
st.set_page_config(layout="wide")

data_folder_path = "Data"

# Load data directly from the folder
for file_name in os.listdir(Path(data_folder_path)):
    if file_name.endswith(".csv"):
        data_file_path = os.path.join(data_folder_path, file_name)
        df = pd.read_csv(data_file_path)
        break  
    
# Load the first CSV file found in the folder
# Loading the data and saved model
# data_file_path = "D:/Project/Customer-Churn/Data/telecom_churn.csv"
# Load data directly from the file
# df = pd.read_csv(data_file_path)

loaded_model1=pickle.load(open('rf1_model.sav','rb'))

# Function to check DataFrame properties
def check(df):
    l=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        duplicated=df.duplicated().sum()
        sum_null=df[col].isnull().sum()
        l.append([col,dtypes,nunique,duplicated,sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['columns','Data Types','No of Unique Values','No of Duplicated Rows','No of Null Values']
    return df_check 

def purchase_pred(input_data):
    input_data_as_array=np.asarray(input_data,dtype=float)
    input_data_reshaped=input_data_as_array.reshape(1,-1)
    prediction=loaded_model1.predict(input_data_reshaped)
    print(prediction)
    
    if prediction[0] == 0:
        return "The forecast indicates that the person will not churn."
    else:
        return "The forecast suggests that the person will churn."

# Figure 1
churn_counts=df["Churn"].value_counts().reset_index()
churn_counts.columns = ['Churn', 'Count']
fig1 = px.bar(churn_counts, x="Churn", y="Count", color='Churn', 
              color_discrete_sequence=['#636EFA', '#EF553B'], 
              labels={'Churn': 'Churn', 'Count': 'Count'}, 
              text='Count', template='plotly', 
              width=600, height=400)

fig1.update_layout(
    xaxis_title="Churn",            # X-axis label
    yaxis_title="Count",            # Y-axis label
    legend_title="Churn",           # Legend title
    title_x=0.5,                    # Title alignment
    title_font=dict(size=20),       
)

contract_counts = df['ContractRenewal'].value_counts()
data_plan_counts = df['DataPlan'].value_counts()
fig2= make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], 
                    subplot_titles=("Contract Renewal", "Data Plan"))
fig2.add_trace(go.Pie(labels=contract_counts.index.map({0: 'Not Renewed', 1: 'Renewed'}), 
                    values=contract_counts.values, name="Contract Renewal"),1, 1)
fig2.add_trace(go.Pie(labels=data_plan_counts.index.map({0: 'No Data Plan', 1: 'Has Data Plan'}), 
                    values=data_plan_counts.values, name="Data Plan"),1, 2)
fig2.update_traces(hole=.4, hoverinfo="label+percent+name")

# Figure 3
nominal_cols=['Churn','DataPlan','ContractRenewal']
num_cols=['AccountWeeks','DataUsage','DayMins','DayCalls','MonthlyCharge','OverageFee','RoamMins']



# Streamlit sidebar
with st.sidebar:
    with st.sidebar.expander(":Red[Info About the Web App]"):
        st.info("This Customer Churn Prediction Web Application empowers users to dive deep into their data, create meaningful visualizations, and leverage advanced machine learning models to predict customer churn. Designed with user-friendliness and interactivity in mind, this web application offers a seamless experience for both data exploration and predictive analytics.")

    # st.image("https://images.datacamp.com/image/upload/v1648487930/shutterstock_1624376548_b831bdf4c1.jpg")
    st.write("---")
    selected = option_menu('Main Menu',
                        ['Data Exploration',
                        'Visualisation',
                        'Prediction'],
                        icons=['search-heart','bar-chart','code'],
                        default_index=0)
    
with st.sidebar:

    # st.title("Main Menu")
    # choice = st.radio("Navigation", ["Data Exploration", "Plots", "ML", "Download"])
    st.divider()
    st.markdown(
        '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/naveen3830"> @naveen</a></h6>',
            unsafe_allow_html=True,
        )


if selected == "Data Exploration": 
    st.markdown("<h1 style='text-align: center;'>Customer Churn Analysis 💼📉</h1>", unsafe_allow_html=True)
    st.divider()
    st.markdown("Customer churn is defined as when customers or subscribers discontinue doing business with a firm or service.Customers in the telecom industry can choose from a variety of service providers and actively switch from one to the next. The telecommunications business has an annual churn rate of 15-25 percent in this highly competitive market.")
    st.markdown("Individualized customer retention is tough because most firms have a large number of customers and can't afford to devote much time to each of them. The costs would be too great, outweighing the additional revenue. However, if a corporation could forecast which customers are likely to leave ahead of time, it could focus customer retention efforts only on these high risk clients. The ultimate goal is to expand its coverage area and retrieve more customers loyalty. The core to succeed in this market lies in the customer itself.")
    
    st.divider()
# Centered Image
    st.markdown(
    "<div style='text-align:center'><img src='https://daxg39y63pxwu.cloudfront.net/images/blog/churn-models/Customer_Churn_Prediction_Models_in_Machine_Learning.png' width='700'></div>",
    unsafe_allow_html=True
    )
    
    st.divider()

    option = st.selectbox("Select an option:", ["Show data in table", "Display data description", "Show dataset dimensions", "Verify data integrity", "Summarize numerical data statistics", "Summarize categorical data"])

    if option == "Show data in table":
            st.data_editor(df)
        
    elif option == "Display data description":
            st.markdown("""
        - **Churn**: Indicates whether the customer churned (Yes or No).
        - **AccountWeeks**: Number of weeks the customer has been with the company.
        - **ContractRenewal**: Whether the customer renewed their contract (1 = Yes, 0 = No).
        - **DataPlan**: Whether the customer has a data plan (1 = Yes, 0 = No).
        - **DataUsage**: Amount of data usage by the customer.
        - **CustServCalls**: Number of customer service calls made by the customer.
        - **DayMins**: Total number of daytime minutes used by the customer.
        - **DayCalls**: Total number of daytime calls made by the customer.
        - **MonthlyCharge**: Monthly charge incurred by the customer.
        - **OverageFee**: Overage fee charged to the customer for exceeding their plan limits.
        - **RoamMins**: Total number of roaming minutes used by the customer.
        """)

    elif option == "Show dataset dimensions":
        shape = f"There are  {df.shape[0]} number of rows and {df.shape[1]} columns in the dataset"
        st.write(shape)

    elif option == "Verify data integrity":
        df_check = check(df)
        st.dataframe(df_check)
        
    elif option == "Summarize numerical data statistics":
        des1 = df.describe().T
        st.dataframe(des1)
                
    elif option == "Summarize categorical data":
        categorical_df = df.select_dtypes(include=['object'])
        if categorical_df.empty:
            st.write("No categorical columns found.")
        else:
            des2 = categorical_df.describe()
            st.dataframe(des2)


if selected == "Visualisation":
    st.markdown("<h1 style='text-align: center;'>Data Visualisation</h1>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<h2 style='text-align: center;'>Custom Scatter Plot</h2>", unsafe_allow_html=True)
    col1,col2=st.columns([0.25,0.75])
    with col1:
        x_axis=st.selectbox('x-axis:',num_cols,index=0)
        y_axis=st.selectbox('y-axis:',num_cols,index=1)
        c_axis=st.selectbox('color',nominal_cols)
    with col2:
        fig3,ax=plt.subplots()
        fig3 = px.scatter(df, x=x_axis, y=y_axis, color=c_axis)
        fig3.update_layout(font=dict(family='Arial',size=12,color='black'),legend=dict(title='Legend Title',font=dict(family='Arial',size=10,color='black')))
        st.plotly_chart(fig3)
    
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        pass

    with col2:
        st.markdown("<h2 style='text-align: center;'>Count Plot Representing the Churn Counts</h2>", unsafe_allow_html=True)
        st.plotly_chart(fig1)
        
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        pass

    with col2:
        st.markdown("<h2 style='text-align: center;'>Pie Chart Representing the Distribution of Contract Renewal and Data Plan</h2>", unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)


    
if selected == "Prediction":
    def main():
        st.markdown("<h1 style='text-align: center;'> Customer Churn Prediction Web App 💼📉</h1>", unsafe_allow_html=True)
        
        col1,col2,col3=st.columns(3)
        # Getting the input data
        with col1:
            AccountWeeks = st.text_input("AccountWeeks")
            
        with col2:
            ContractRenewal = st.text_input("ContractRenewal")
        
        with col3:
            DataPlan = st.text_input("DataPlan")
        
        with col1:
            DataUsage = st.text_input("DataUsage")
        
        with col2:
            CustServCalls = st.text_input("CustServCalls")
        
        with col3:
            DayMins = st.text_input("DayMins")
        
        with col1:
            DayCalls = st.text_input("DayCalls")
        
        with col2:
            MonthlyCharge = st.text_input("MonthlyCharge")
        
        with col3:
            OverageFee = st.text_input("OverageFee")
            
        with col1:
            RoamMins = st.text_input("RoamMins")
    
        # Code for prediction
        Purchased = ''
    
        if st.button("Result"):
            input_data = [AccountWeeks, ContractRenewal, DataPlan, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins]
            Purchased = purchase_pred(input_data)
    
        st.success(Purchased)

    if __name__ == '__main__':
        main()


if selected == "Download":
    pass

