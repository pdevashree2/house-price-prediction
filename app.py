import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title("🏠 House Price Prediction")
st.write("California Housing Dataset — Machine Learning App")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    df = pd.get_dummies(df, columns=['ocean_proximity'])
    return df

df = load_data()
st.subheader("Raw Data")
st.dataframe(df.head())

# Train model
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
st.subheader("Feature Importance")
importances = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots(figsize=(10,6))
importances.sort_values().plot(kind='barh', ax=ax)
st.pyplot(fig)

st.success(f"Model R² Score: {model.score(X_test, y_test):.4f}")