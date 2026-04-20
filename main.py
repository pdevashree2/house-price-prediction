import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# Quick look
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
df['median_house_value'].hist(bins=50, figsize=(10,5))
plt.xlabel('House Value')
plt.ylabel('Count')
plt.title('Distribution of House Prices')
plt.show()
df.hist(bins=50, figsize=(15,10))
plt.tight_layout()
plt.show()
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
        s=df['population']/100, c='median_house_value',
        cmap='jet', colorbar=True, figsize=(10,7))
plt.title('House Prices by Location')
plt.show()
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix['median_house_value'].sort_values(ascending=False))
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# Fill missing values with the median
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

# Verify no more missing values
print("Missing values:\n", df.isnull().sum())
df = pd.get_dummies(df, columns=['ocean_proximity'])
print(df.shape)
print(df.head())


#mlmodel
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print(f"RMSE: {rmse:,.0f}")
print(f"R² Score: {r2:.4f}")

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest Results:")
print(f"RMSE: {rf_rmse:,.0f}")
print(f"R² Score: {rf_r2:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, rf_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(10,6))
plt.title('Feature Importance')
plt.show()

results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': rf_pred
})
results.to_csv('predictions.csv', index=False)
print("Results saved to predictions.csv!")