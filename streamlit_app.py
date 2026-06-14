import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder

@st.cache_resource
def load_model():
    return joblib.load("fraud_model_NEW.pkl")

model = load_model()



st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_parquet("dashboard_data.parquet")

df = load_data()

# ==================== CUSTOM STYLE ====================

st.markdown("""
<style>
.stMultiSelect [data-baseweb="tag"]{
    background-color:#4F46E5 !important;
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================

st.sidebar.title("Filters")

page = st.sidebar.radio(
    "Navigation",
    ["Dataset Info", "EDA", "Analysis", "Prediction"]
)

selected_months = st.sidebar.multiselect(
    "Transaction Month",
    sorted(df['transaction_month'].unique()),
    default=sorted(df['transaction_month'].unique())
)

selected_categories = st.sidebar.multiselect(
    "Transaction Category",
    df['category'].unique(),
    default=df['category'].unique()
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    df['gender'].unique(),
    default=df['gender'].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['age'].min()),
    max_value=int(df['age'].max()),
    value=(
        int(df['age'].min()),
        int(df['age'].max())
    )
)

# ==================== APPLY FILTERS ====================

filtered = df[
    (df['transaction_month'].isin(selected_months))
    & (df['category'].isin(selected_categories))
    & (df['gender'].isin(selected_gender))
    & (df['age'].between(age_range[0], age_range[1]))
]

# ======================================================
# DATASET INFO
# ======================================================

if page == "Dataset Info":

    st.title("💳 Credit Card Fraud Detection Dashboard")

    st.markdown("""
    ### About the Dataset

    This dataset contains credit card transactions used to identify fraudulent activities.

    The data includes transaction details, customer demographics, merchant information,
    and engineered behavioral features used for fraud detection.

    ### Business Problem

    Credit card fraud causes significant financial losses to both banks and customers.

    Detecting fraudulent transactions quickly can help:

    - Reduce financial losses
    - Improve customer trust
    - Enhance security systems
    - Support risk management decisions

    ### Project Goal

    Build a machine learning model capable of accurately classifying
    fraudulent and legitimate transactions.
    """)

    st.divider()

    total_transactions = len(filtered)
    fraud_transactions = filtered['is_fraud'].sum()
    fraud_rate = round(filtered['is_fraud'].mean() * 100, 2)
    avg_amount = round(filtered['amt'].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transactions", f"{total_transactions:,}")
    col2.metric("Fraud Cases", f"{fraud_transactions:,}")
    col3.metric("Fraud Rate", f"{fraud_rate}%")
    col4.metric("Average Amount", f"${avg_amount}")

    st.divider()

    st.subheader("Dataset Shape")

    c1, c2 = st.columns(2)

    c1.metric("Rows", f"{filtered.shape[0]:,}")
    c2.metric("Columns", filtered.shape[1])

    st.divider()

    st.subheader("Data Preview")

    st.dataframe(
        filtered.head(20),
        use_container_width=True
    )

    st.divider()

    st.subheader("Column Information")

    info_df = pd.DataFrame({
        "Column": filtered.columns,
        "Data Type": filtered.dtypes.astype(str)
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

# ======================================================
# EDA
# ======================================================

elif page == "EDA":

    st.title("📉 Exploratory Data Analysis (EDA)")

    st.markdown("""
    ### 🔍 Understanding Transaction Patterns

    Exploratory Data Analysis helps uncover patterns, trends, and anomalies within credit card transactions.

    Through visualizations, we can identify:

    - Customer behavior patterns
    - Transaction amount distributions
    - Fraud occurrence trends
    - Spending behavior across categories
    - Demographic characteristics

    These insights support fraud detection and risk management.
    """)

    st.subheader("📊 Statistical Summary")

    st.dataframe(filtered.describe())

    st.divider()

    # ==================== Transaction Amount ====================

    st.subheader("💰 Distribution of Transaction Amounts")

    fig1 = px.histogram(
        filtered,
        x="amt",
        nbins=30,
        title="Distribution of Transaction Amounts"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    💡 **Insight:** Most transactions are concentrated at lower amounts, while a small number of transactions have very high values. This right-skewed distribution suggests the presence of outliers and unusual spending patterns.
    """)

    st.divider()

    # ==================== Age Distribution ====================

    st.subheader("👥 Distribution of Customer Age")

    fig2 = px.histogram(
        filtered,
        x="age",
        nbins=30,
        title="Distribution of Age"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    💡 **Insight:** The majority of customers are between 30 and 60 years old. Very young and very old customers represent a smaller proportion of transactions.
    """)

    st.divider()

    # ==================== Distance ====================

    st.subheader("📍 Distribution of Customer-Merchant Distance")

    fig3 = px.histogram(
        filtered,
        x="distance",
        title="Distribution of Distance"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    💡 **Insight:** Most transactions occur within moderate distances from the customer's location. Extremely short and extremely long distances are less frequent and may indicate unusual behavior.
    """)

    st.divider()

    # ==================== Fraud vs Non Fraud ====================

    st.subheader("🚨 Fraud vs Non-Fraud Transactions")

    fraud_counts = (
        filtered["is_fraud"]
        .value_counts()
        .reset_index(name="count")
    )

    fraud_counts.columns = ["is_fraud", "count"]

    fig4 = px.bar(
        fraud_counts,
        x="is_fraud",
        y="count",
        color="is_fraud",
        title="Fraud vs Non-Fraud"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    💡 **Insight:** Fraudulent transactions represent only a very small percentage of the dataset. This indicates a highly imbalanced classification problem.
    """)

    st.divider()

    # ==================== Categories ====================

    st.subheader("🛒 Transaction Categories")

    category_counts = (
        filtered["category"]
        .value_counts()
        .reset_index(name="count")
    )

    category_counts.columns = ["category", "count"]

    fig5 = px.bar(
        category_counts,
        x="category",
        y="count",
        title="Transaction Categories"
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    💡 **Insight:** Categories such as gas_transport, grocery_pos, and home generate the highest transaction volumes, indicating where customers spend most frequently.
    """)

    st.divider()

    # ==================== Gender ====================

    st.subheader("Gender Distribution")

    gender_counts = (
        filtered["gender"]
        .value_counts()
        .reset_index(name="count")
    )

    gender_counts.columns = ["gender", "count"]

    fig6 = px.bar(
        gender_counts,
        x="gender",
        y="count",
        title="Gender Distribution"
    )

    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("""
    💡 **Insight:** Female customers account for slightly more transactions than male customers. The difference is relatively small, indicating a balanced customer base.
    """)

elif page == "Analysis":

    st.title("📊 Business Analysis")

    # ==========================================================
    # Question 1
    # ==========================================================

    with st.expander("📌Question 1: Does Transaction Amount Affect Fraud?"):

        st.write("### 📈 Visualization")

        fig1 = px.box(
            df,
            x='is_fraud',
            y='amt',
            title='Amount vs Fraud'
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.write("### 💡 Insight")

        st.info("""
 💡 Insight:

    Most transactions, whether fraudulent or non fraudulent , occur at relatively low transaction amounts.

    The boxplot shows that non fraudulent transactions contain a larger number of extreme high-value outliers compared to fraudulent transactions. While fraudulent transactions are present across different transaction amounts, they do not necessarily correspond to the highest transaction values.

    This suggests that transaction amount alone is not sufficient to identify fraud and should be combined with other behavioral and transactional features for accurate fraud detection.
        """)

    # ==========================================================
    # Question 2
    # ==========================================================

    with st.expander("📌 Question 2: Do fraudulent transactions occur at larger distances?"):

        st.write("### 📈 Visualization")

        fig2 = px.box(
            df,
            x='is_fraud',
            y='distance',
            title='Distance vs Fraud'
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.write("### 💡 Insight")

        st.info("""
        The distribution of transaction distances is very similar for both fraudulent and legitimate transactions.

        The median distance and interquartile range (IQR) for fraud and non-fraud transactions are nearly identical, indicating that fraudulent transactions do not necessarily occur at significantly larger distances.

        This suggests that distance alone may not be a strong predictor of fraud. However, when combined with other features such as transaction amount, customer behavior, or merchant information, it may still contribute to improving fraud detection performance.
        """)

    # ==========================================================
    # Question 3
    # ==========================================================

    with st.expander("📌 Question 3: At what hours does fraud occur most frequently?"):

        st.write("### 📈 Visualization")

        hour_labels = {
            0:'12 AM', 1:'1 AM', 2:'2 AM', 3:'3 AM',
            4:'4 AM', 5:'5 AM', 6:'6 AM', 7:'7 AM',
            8:'8 AM', 9:'9 AM', 10:'10 AM', 11:'11 AM',
            12:'12 PM', 13:'1 PM', 14:'2 PM', 15:'3 PM',
            16:'4 PM', 17:'5 PM', 18:'6 PM', 19:'7 PM',
            20:'8 PM', 21:'9 PM', 22:'10 PM', 23:'11 PM'
        }

        fraud_hour = (
            df[df['is_fraud'] == 1]['transaction_hour']
            .map(hour_labels)
            .value_counts()
        )

        fig3 = px.bar(
            x=fraud_hour.index,
            y=fraud_hour.values,
            labels={'x': 'Hour', 'y': 'Fraud Count'},
            title='Fraud Cases by Hour'
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.write("### 💡 Insight")

        st.info("""
        Fraudulent transactions are highly concentrated during late-night hours, particularly around 10 PM and 11 PM.

        The number of fraud cases drops significantly during daytime hours, suggesting that fraudsters may prefer operating at night when customer activity monitoring is lower and suspicious transactions are less likely to be noticed immediately.

        This pattern indicates that transaction time is a valuable feature for fraud detection models.
        """)

    # ==========================================================
    # Question 4
    # ==========================================================

    with st.expander("📌 Question 4: Does fraud vary by day of the week?"):

        st.write("### 📈 Visualization")

        day_names = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thursday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }

        fraud_days = (
            df[df['is_fraud'] == 1]['transaction_dayofweek']
            .map(day_names)
            .value_counts()
            .sort_values(ascending=False)
        )

        fig4 = px.bar(
            x=fraud_days.index,
            y=fraud_days.values,
            labels={'x': 'Day of Week', 'y': 'Fraud Count'},
            title='Fraud Cases by Day of Week'
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.write("### 💡 Insight")

        st.info("""
        Fraudulent transactions occur throughout the week, but the highest number of fraud cases is observed during weekends, particularly on Saturday and Sunday.

        Weekdays such as Tuesday and Wednesday show relatively lower fraud activity. This may indicate that fraudsters take advantage of increased transaction activity during weekends.

        The day of the week appears to provide useful information when identifying fraud patterns.
        """)

    # ==========================================================
    # Question 5
    # ==========================================================

    with st.expander("📌 Question 5: Which states experience the highest number of fraudulent transactions?"):

        st.write("### 📈 Visualization")

        fraud_states = (
            df.groupby('state')['is_fraud']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig5 = px.bar(
            x=fraud_states.index,
            y=fraud_states.values,
            labels={'x': 'State', 'y': 'Fraud Count'},
            title='Top 10 States by Fraud Count'
        )

        st.plotly_chart(fig5, use_container_width=True)

        st.write("### 💡 Insight")

        st.info("""
        Fraudulent transactions are not distributed evenly across states. New York (NY) records the highest number of fraud cases, followed by Texas (TX) and Pennsylvania (PA).

        Geographic location appears to influence fraud patterns, although differences in population size and transaction volume should also be considered.

        Location-based features can therefore provide valuable information for fraud detection models.
        """)

    # ==========================================================
    # Question 6
    # ==========================================================

    with st.expander("📌 Question 6: Is fraud more common among certain age groups?"):

        st.write("### 📈 Visualization")

        fig6 = px.box(
            df,
            x='is_fraud',
            y='age',
            title='Age vs Fraud'
        )

        st.plotly_chart(fig6, use_container_width=True)

        st.write("### 💡 Insight")

        st.info("""
        Fraud cases occur across a wide range of age groups, and there is considerable overlap between fraudulent and legitimate transactions.

        Although fraudulent transactions appear to involve slightly older customers on average, age alone is not a strong indicator of fraud.

        Age becomes more useful when combined with other transaction and customer-related features.
        """)

    with st.expander("📌 Question 7: Are some transaction categories more associated with fraud?"):

     category_fraud = (
        filtered[filtered["is_fraud"] == 1]
        .groupby("category")
        .size()
        .reset_index(name="fraud_count")
        .sort_values("fraud_count", ascending=False)
    )

    fig = px.bar(
        category_fraud,
        x="category",
        y="fraud_count",
        title="Fraud Transactions by Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    💡 **Insight:**

  Grocery , online shopping transactions appear to experience higher levels of fraudulent activity compared to many in-store categories.
  This may be due to the increased risk associated with card-not-present transactions, where stolen card information can be used more easily
    """) 

    # ==========================================================
    # Key Findings
    # ==========================================================

    st.subheader("📌 Key Findings")

    st.success("""
    • Fraud activity differs across transaction categories.

    • Transaction amount shows noticeable differences between fraudulent and legitimate transactions.

    • Fraud occurs more frequently during late-night hours.

    • Weekend days experience slightly higher fraud activity than weekdays.

    • Certain states report higher fraud counts than others.

    • Age alone is not a strong fraud indicator, but it can contribute when combined with other features.

    • Transaction category, amount, time, location, and customer characteristics are valuable predictors for fraud detection.
    """)

#----------------------
elif page == "Prediction":

    st.title("💳 Credit Card Fraud Prediction")

    st.image(
        "https://www.consultancy-me.com/illustrations/news/spotlight/2023-11-23-120124299-fraud_detection_spot.jpg",
        width=1500
    )

    st.markdown("""
    ### Enter Transaction Information

    Fill in the transaction details below and click **Predict** to determine whether the transaction is fraudulent or legitimate.
    """)

    col1, col2 = st.columns(2)

    with col1:

        category = st.selectbox(
            "Category",
            sorted(df["category"].unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["gender"].unique())
        )

        state = st.selectbox(
            "State",
            sorted(df["state"].unique())
        )

        amt = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

        city_pop = st.number_input(
            "City Pop",
            min_value=0,
            value=50000
        )

        merch_zipcode = st.number_input(
            "Merchant Zipcode",
            min_value=0,
            value=10000
        )

    with col2:

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

        distance = st.number_input(
            "Distance",
            min_value=0.0,
            value=10.0
        )

        transaction_year = st.number_input(
            "Transaction Year",
            value=2020
        )

        transaction_month = st.slider(
            "Month",
            1,
            12,
            6
        )

        transaction_hour = st.slider(
            "Hour",
            0,
            23,
            12
        )

        transaction_dayofweek = st.slider(
            "Day Of Week",
            0,
            6,
            3
        )

        transaction_day = st.slider(
            "Day",
            1,
            31,
            15
        )

    st.markdown("---")

    if st.button("🔍 Predict Fraud"):

     input_data = pd.DataFrame({
        "category": [category],
        "amt": [amt],
        "gender": [gender],
        "state": [state],
        "city_pop": [city_pop],
        "merch_zipcode": [merch_zipcode],
        "age": [age],
        "distance": [distance],
        "transaction_year": [transaction_year],
        "transaction_month": [transaction_month],
        "transaction_hour": [transaction_hour],
        "transaction_dayofweek": [transaction_dayofweek],
        "transaction_day": [transaction_day]
    })

     prediction = model.predict(input_data)[0]

     if prediction == 1:
        st.error("🚨 Fraud Detected")
     else:
        st.success("✅ No Fraud Detected")
