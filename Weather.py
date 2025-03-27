import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Streamlit UI
st.title("Future Temperature Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display dataset preview
    st.write("## Dataset Preview")
    st.write(df.head())

    # Ensure correct column names
    df.columns = df.columns.str.strip()  # Remove extra spaces

    # Check if "YEAR" column exists
    if "YEAR" not in df.columns:
        st.error("YEAR column is missing! Please check the dataset format.")
    else:
        # Define features (Years) and target variables (Monthly Temperatures)
        X = df[["YEAR"]]  # Independent variable (Years)
        y = df.drop(columns=["YEAR"])  # Dependent variables (Temperature values for each month)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict future temperatures (e.g., next 5 years)
        future_years = np.arange(df["YEAR"].max() + 1, df["YEAR"].max() + 6).reshape(-1, 1)
        future_temps = model.predict(future_years)

        # Convert predictions to DataFrame
        future_df = pd.DataFrame(future_temps, columns=y.columns, index=future_years.flatten())

        # Display future predictions
        st.write("## Predicted Future Temperatures (Next 5 Years)")
        st.write(future_df)

        # Visualization
        st.write("## Temperature Trend Visualization")
        fig, ax = plt.subplots(figsize=(10, 5))
        for month in y.columns:
            sns.lineplot(x=df["YEAR"], y=df[month], ax=ax, label=month, alpha=0.5)
            sns.lineplot(x=future_df.index, y=future_df[month], ax=ax, linestyle="dashed")

        plt.xlabel("Year")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Trends & Future Predictions")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)
