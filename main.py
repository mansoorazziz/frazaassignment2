# # ======================================================================================================================
# # ******************************** Condition Monitoring and Sensor Data Analysis ***************************************
# # ======================================================================================================================

import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# # ======================================================================================================================
# # ======================================================================================================================

# Start time
start_time = time.time()


# # -------------------------------1. Load the dataset ----------------------------------------------
# # 
# # 
rawData = pd.read_csv(
    'data.csv',
    na_values='NA',  # Treat 'NA' as missing values
)


# # 
print(rawData.tail(10))

# # ------------------------------- 2. Check for missing values in each column ----------------------------------------------

# # 
missing_values = rawData.isnull().sum()
print("Missing values in each column:")
print(missing_values)


# # # -------------------------------3. Handling Missing Values ------------------------------------------------------------

# Fill numerical columns with their mean
numerical_columns = ['footfall', 'tempMode', 'AQ', 'USS', 'CS']
# Convert non-numeric columns to numeric (if necessary)
for col in numerical_columns:
    rawData[col] = pd.to_numeric(rawData[col], errors='coerce')
rawData[numerical_columns] = rawData[numerical_columns].fillna(rawData[numerical_columns].mean())


# # -------------------------------4. Normalizing dataset Using Min-Max Scaler  ----------------------------------------------------------

# Columns to needs tp normalize
columns_to_normalize = ['footfall', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP']

# Apply Min-Max Scaling
scaler = MinMaxScaler()
rawData[columns_to_normalize] = scaler.fit_transform(rawData[columns_to_normalize])
print(rawData.head())


# ---------------------------5. Computing the correlations for finding good features ------------------------------------------

correlation_matrix = rawData.corr(method='pearson')
fail_correlation = correlation_matrix['fail'].sort_values(ascending=False)


# ---------------------------6. Visualisation as a Bar chart ------------------------------------------

fail_correlation[:-1].plot(kind='bar', color='skyblue', figsize=(10, 6), title='Feature Correlation with fail')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.show()



# # ======================================================================================================================
# # ************************************************////Feature Engineering/////////**************************************
# # ======================================================================================================================


# -------------------------------7. Calculating Interaction Features  -----------------------------------------------------

# Between Temprature and RP
rawData['Diff_TemptoRp'] = rawData['Temperature'] - rawData['RP']

# # Between Air quality and RP
rawData['Diff_UsstoAq'] = rawData['AQ'] - rawData['USS']


# ---------------------------------------------8. Feature selection  -------------------------------------------------------
 
selected_features = ['VOC', 'AQ', 'USS', 'Temperature', 'Diff_TemptoRp', 'Diff_UsstoAq']
X = rawData[selected_features]
y = rawData['fail']

# ---------------------------------------------9. Train Test Split  -------------------------------------------------------
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------10. Model training  -------------------------------------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)



# ---------------------------------------------11. Evaluating a Modal  -------------------------------------------------------


y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# End time
end_time = time.time()

# Calculate runtime
print(f"Total runtime: {end_time - start_time} seconds")

# # ======================================================================================================================
# # ************************************************ END ******************************************
# # ======================================================================================================================


