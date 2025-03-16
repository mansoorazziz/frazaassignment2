# # ======================================================================================================================
# # ************************************************  *************************************************************
# # ======================================================================================================================

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# # ======================================================================================================================
# # ************************************************ PART- I ***********************************************************
# # ======================================================================================================================


# # -------------------------------Load the dataset ----------------------------------------------
# # 
# # 
rawData = pd.read_csv(
    'data.csv',
    na_values='NA',  # Treat 'NA' as missing values
)

# # -------------------------------Print the last 10 rows of the dataset -------------------------------------------------

# # 
print(rawData.tail(10))

# # ------------------------------- Check for missing values in each column ----------------------------------------------

# #
missing_values = rawData.isnull().sum()
print("Missing values in each column:")
print(missing_values)





# # # ======================================================================================================================
# # # ************************************************ PART- II ***********************************************************
# # # ======================================================================================================================


# # # -------------------------------Handling Missing Values ------------------------------------------------------------

# Step 1: Handle missing values
# Fill numerical columns with their mean
numerical_columns = ['footfall', 'tempMode', 'AQ', 'USS', 'CS']
# Convert non-numeric columns to numeric (if necessary)
for col in numerical_columns:
    rawData[col] = pd.to_numeric(rawData[col], errors='coerce')
rawData[numerical_columns] = rawData[numerical_columns].fillna(rawData[numerical_columns].mean())


# # -------------------------------2. Standardized_dataset Using Min-Max Scaler  ----------------------------------------------------------

# Select columns to normalize
columns_to_normalize = ['footfall', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP']

# Apply Min-Max Scaling
scaler = MinMaxScaler()
rawData[columns_to_normalize] = scaler.fit_transform(rawData[columns_to_normalize])

# print(rawData.head())
# # # ======================================================================================================================
# # # ************************************************ PART- III ***********************************************************
# # # ======================================================================================================================

# ---------------------------7. Computing the correlations for finding good features ------------------------------------------



correlation_matrix = rawData.corr(method='pearson')
# # Create a heatmap
# p# Correlation of each feature with 'fail'
fail_correlation = correlation_matrix['fail'].sort_values(ascending=False)

# Visualize as a bar chart
fail_correlation[:-1].plot(kind='bar', color='skyblue', figsize=(10, 6), title='Feature Correlation with fail')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.show()



# # ======================================================================================================================
# # ************************************************ PART- IV    Feature Engineering**************************************
# # ======================================================================================================================


# -------------------------------3. Calculating Interaction Features between Temperature and RP ----------------------------------------------------------
rawData['Diff_TemptoRp'] = rawData['Temperature'] - rawData['RP']

# -------------------------------3. Calculating Interaction Features between Temperature and RP ----------------------------------------------------------
rawData['Diff_UsstoAq'] = rawData['AQ'] - rawData['USS']
# print(rawData.head())

# Step 2: Feature selection
selected_features = ['VOC', 'AQ', 'USS', 'Temperature', 'Diff_TemptoRp', 'Diff_UsstoAq']
X = rawData[selected_features]
y = rawData['fail']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 4: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


# Step 6: Model evaluation
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
# # ======================================================================================================================
# # ************************************************ PART- V  Data Visualization******************************************
# # ======================================================================================================================


