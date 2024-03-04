import streamlit as st, pandas as pd, numpy as np, matplotlib as mlp, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, plotly.graph_objs as go, plotly.express as ex, plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
from scipy.stats import chi2
import squarify
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.decomposition import PCA



df = pd.read_excel('E Commerce Dataset.xlsx')

# load the first 5 rows of the dataset
#display column variables
st.write("Column Variables:", df.columns.tolist())

#Display the Dataframe
st.write("E Commerce Dataset.xlsx:", df)



#Check for duplicate
duplicates = df[df.duplicated(keep=False)]
st.subheader('Duplicate Records:')
st.write(duplicates)

#Check for missing values
missing_values = df.isnull().sum()

#Display missing values
st.write("Missing Values:")
st.write(missing_values)

#Display the Dataframe
st.write("E Commerce Dataset.xlsx:", df)

# Replace missing values with median
for col in df.columns:
    if df[col].dtype != object:  # Exclude non-numeric columns
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)

# Display the DataFrame with missing values replaced
st.write("Data with Missing Values Replaced by Median:")
st.write(df)

# Check for duplicate values within each category
#duplicate_values = df[df.duplicated(subset=['CustomerID'], keep=False)]

# Display duplicate values within each category
#st.write("Duplicate Values within each Category:")
#st.write(duplicate_values)

# Function to check for duplicate values within each category
def check_duplicates_by_category(df):
    duplicate_values_by_category = {}
    for col in df.columns:
        if col != 'Category':
            continue
        unique_categories = df[col].unique()
        for category in unique_categories:
            subset_df = df[df[col] == category]
            duplicate_values = subset_df[subset_df.duplicated(subset=['CustomerID'], keep=False)]
            if not duplicate_values.empty:
                if category not in duplicate_values_by_category:
                    duplicate_values_by_category[category] = duplicate_values
                else:
                    duplicate_values_by_category[category] = pd.concat([duplicate_values_by_category[category], duplicate_values])
    return duplicate_values_by_category

# Check for duplicate values within each category
duplicate_values_by_category = check_duplicates_by_category(df)

# Display duplicate values within each category
st.write("Duplicate Values within each Category:")
for category, duplicate_values in duplicate_values_by_category.items():
    st.write(f"Category: {category}")
    st.write(duplicate_values)


# Title for the dashboard
st.title('Ecommerce Churn Analysis Dashboard')

# Sidebar for filtering options
st.sidebar.title('Filter Options')

# Sidebar options
analysis_option = st.sidebar.selectbox('Select Analysis', ['Customer Churn Rate', 'Order Churn Rate'])

# Main content based on selected analysis option
if analysis_option == 'Customer Churn Rate':
    # Calculate customer churn rate
    total_customers = df['CustomerID'].nunique()
    churned_customers = df[df['Churn'] == 1]['CustomerID'].nunique()
    churn_rate = churned_customers / total_customers

    # Display customer churn rate
    st.subheader('Customer Churn Rate')
    st.write(f"Total Customers: {total_customers}")
    st.write(f"Churned Customers: {churned_customers}")
    st.write(f"Churn Rate: {churn_rate:.2%}")

elif analysis_option == 'Order Churn Rate':
    # Calculate product churn rate
    total_products = df['OrderCount'].nunique()
    churned_products = df[df['Churn'] == 1]['OrderCount'].nunique()
    churn_rate = churned_products / total_products

    # Display product churn rate
    st.subheader('Order Churn Rate')
    st.write(f"Total products: {total_products}")
    st.write(f"Churned products: {churned_products}")
    st.write(f"Churn Rate: {churn_rate:.2%}")



# Load ecommerce data
@st.cache_data
def load_data():
    return pd.read_excel('E Commerce Dataset.xlsx')

df = load_data()

# Title for the dashboard
st.title('Churn Analysis')

# Calculate churn rate
total_customers = df['CustomerID'].nunique()
churned_customers = df[df['Churn'] == 1]['CustomerID'].nunique()
churn_rate = churned_customers / total_customers

# Create a pie chart for churn customers
fig, ax = plt.subplots()
ax.pie([churned_customers, total_customers - churned_customers], labels=['Churn', 'Non-Churn'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
st.pyplot(fig)

# Display churn rate
st.write(f"Churn Rate: {churn_rate:.2%}")

# Load ecommerce data
@st.cache_data
def load_data():
    return pd.read_excel('E Commerce Dataset.xlsx')

df = load_data()

# Title for the dashboard
st.title('Customer Churn Analysis by Gender')

# Calculate churn rate by gender
gender_churn_counts = df.groupby('Gender')['Churn'].sum()
gender_total_counts = df['Gender'].value_counts()
gender_churn_rate = gender_churn_counts / gender_total_counts

# Define colors for the bars
colors = ['blue', 'orange']

# Create a bar chart for churn rate by gender
fig, ax = plt.subplots()
gender_churn_rate.plot(kind='bar', ax=ax, color=colors)
ax.set_ylabel('Churn Rate')
ax.set_xlabel('Gender')

# Display the bar chart
st.pyplot(fig)

# Display churn rate by gender
st.write("Churn Rate by Gender:")
st.write(gender_churn_rate)

# Load ecommerce data
@st.cache_data
def load_data():
    return pd.read_excel('E Commerce Dataset.xlsx')

df = load_data()

# Title for the dashboard
st.title('Distribution of Orders by Customer Churn Status')

# Calculate the number of orders per customer for churned and non-churned customers
churned_customers = df[df['Churn'] == 1]
non_churned_customers = df[df['Churn'] == 0]

orders_per_churned_customer = churned_customers['CustomerID'].value_counts()
orders_per_non_churned_customer = non_churned_customers['CustomerID'].value_counts()

# Calculate the counts of customers with different number of orders for churned and non-churned customers
churned_customer_order_counts = orders_per_churned_customer.value_counts().sort_index()
non_churned_customer_order_counts = orders_per_non_churned_customer.value_counts().sort_index()

# Determine the range of x-axis
max_orders = max(churned_customer_order_counts.index.max(), non_churned_customer_order_counts.index.max())

# Create a dual bar chart for the distribution of orders by churn status
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for churned customers
ax.bar(churned_customer_order_counts.index, churned_customer_order_counts, width=0.4, color='red', label='Churned')

# Plot bars for non-churned customers with a slight offset
ax.bar(non_churned_customer_order_counts.index + 0.4, non_churned_customer_order_counts, width=0.4, color='blue', label='Non-Churned')

ax.set_xlabel('Number of Orders')
ax.set_ylabel('Count of Customers')
ax.set_title('Distribution of Orders by Customer Churn Status')
ax.set_xticks(range(1, max_orders + 1))
ax.legend()

# Display the dual bar chart
st.pyplot(fig)

# Load ecommerce data
@st.cache_data
def load_data():
    return pd.read_excel('E Commerce Dataset.xlsx')

df = load_data()

# Title for the dashboard
st.title('Comparison of Preferred Login Devices for Churned and Non-Churned Customers')

# Calculate counts of preferred login devices for churned and non-churned customers
churned_login_devices = df[df['Churn'] == 1]['PreferredLoginDevice'].value_counts()
non_churned_login_devices = df[df['Churn'] == 0]['PreferredLoginDevice'].value_counts()

# Get unique login devices
unique_login_devices = df['PreferredLoginDevice'].unique()

# Create a grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for churned customers
ax.bar(unique_login_devices, churned_login_devices, width=0.4, color='red', label='Churned')

# Plot bars for non-churned customers with a slight offset
ax.bar([str(x) + ' ' for x in unique_login_devices], non_churned_login_devices, width=0.4, color='blue', label='Non-Churned')

ax.set_xlabel('Login Device')
ax.set_ylabel('Count')
ax.set_title('Comparison of Preferred Login Devices for Churned and Non-Churned Customers')
ax.legend()

# Display the grouped bar chart
st.pyplot(fig)

# Load sample ecommerce data
@st.cache_data
def load_data():
    return pd.read_excel('E Commerce Dataset.xlsx')

df = load_data()

# Title for the dashboard
st.title('Comparison of Preferred Payment Modes for Churned and Non-Churned Customers')

# Calculate counts of preferred payment modes for churned and non-churned customers
churned_payment_modes = df[df['Churn'] == 1]['PreferredPaymentMode'].value_counts()
non_churned_payment_modes = df[df['Churn'] == 0]['PreferredPaymentMode'].value_counts()

# Get unique payment modes
unique_payment_modes = df['PreferredPaymentMode'].unique()

# Create a grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(20, 16))

# Plot bars for churned customers
ax.bar(unique_payment_modes, churned_payment_modes, width=0.4, color='red', label='Churned')

# Plot bars for non-churned customers with a slight offset
ax.bar([str(x) + ' ' for x in unique_payment_modes], non_churned_payment_modes, width=0.4, color='blue', label='Non-Churned')

ax.set_xlabel('Payment Mode')
ax.set_ylabel('Count')
ax.set_title('Comparison of Preferred Payment Modes for Churned and Non-Churned Customers')
ax.legend()

# Display the grouped bar chart
st.pyplot(fig)

# Load ecommerce data
@st.cache_data
def load_data():
    return pd.read_excel('E Commerce Dataset.xlsx')

df = load_data()

# Title for the dashboard
st.title('Comparison of Preferred Order Categories for Churned and Non-Churned Customers')

# Calculate counts of preferred order categories for churned and non-churned customers
churned_order_categories = df[df['Churn'] == 1]['PreferedOrderCat'].value_counts()
non_churned_order_categories = df[df['Churn'] == 0]['PreferedOrderCat'].value_counts()

# Get unique order categories
unique_order_categories = df['PreferedOrderCat'].unique()

# Create a grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(14, 10))

# Plot bars for churned customers
ax.bar(unique_order_categories, churned_order_categories, width=0.4, color='red', label='Churned')

# Plot bars for non-churned customers with a slight offset
ax.bar([str(x) + ' ' for x in unique_order_categories], non_churned_order_categories, width=0.4, color='blue', label='Non-Churned')

ax.set_xlabel('Order Category')
ax.set_ylabel('Count')
ax.set_title('Comparison of Preferred Order Categories for Churned and Non-Churned Customers')
ax.legend()

# Display the grouped bar chart
st.pyplot(fig)

# Load ecommerce data
@st.cache_data
def load_data():
    return pd.read_excel('E Commerce Dataset.xlsx')

df = load_data()

# Title for the dashboard
st.title('Comparison of Customer Distance from Warehouse to Home for Churned and Non-Churned Customers')

# Calculate mean distance from warehouse to home for churned and non-churned customers
churned_distance = df[df['Churn'] == 1]['WarehouseToHome'].mean()
non_churned_distance = df[df['Churn'] == 0]['WarehouseToHome'].mean()

# Create a bar chart for the comparison
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars for churned and non-churned customers
ax.bar(['Churned', 'Non-Churned'], [churned_distance, non_churned_distance], color=['red', 'blue'])

ax.set_xlabel('Customer Churn Status')
ax.set_ylabel('Mean Distance from Warehouse to Home')
ax.set_title('Comparison of Customer Distance from Warehouse to Home for Churned and Non-Churned Customers')

# Display the bar chart
st.pyplot(fig)

# Title for the dashboard
st.title('Comparison of Customer City Tier for Churned and Non-Churned Customers')

# Calculate value counts of city tiers for churned and non-churned customers
churned_city_tiers = df[df['Churn'] == 1]['CityTier'].value_counts()
non_churned_city_tiers = df[df['Churn'] == 0]['CityTier'].value_counts()

# Get unique city tiers
unique_city_tiers = df['CityTier'].unique()

# Create a grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for churned customers
ax.bar(unique_city_tiers, churned_city_tiers, width=0.4, color='red', label='Churned')

# Plot bars for non-churned customers with a slight offset
ax.bar([x + 0.4 for x in unique_city_tiers], non_churned_city_tiers, width=0.4, color='blue', label='Non-Churned')

ax.set_xlabel('City Tier')
ax.set_ylabel('Count')
ax.set_title('Comparison of Customer City Tier for Churned and Non-Churned Customers')
ax.legend()

# Display the grouped bar chart
st.pyplot(fig)


# Title for the dashboard
st.title('Comparison of Hours Spent on App for Churned and Non-Churned Customers')

# Filter out rows with negative values for 'Hours_Spent_On_App' (assuming negative values indicate data issues)
df = df[df['HourSpendOnApp'] >= 0]

# Calculate hours spent on app for churned and non-churned customers
churned_hours = df[df['Churn'] == 1]['HourSpendOnApp']
non_churned_hours = df[df['Churn'] == 0]['HourSpendOnApp']

# Set up bins for the histogram
bins = range(0, int(df['HourSpendOnApp'].max()) + 4, 4)  # Adjust bin size as needed

# Create a grouped histogram for the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram for churned customers
ax.hist(churned_hours, bins=bins, color='red', alpha=0.5, label='Churned', edgecolor='black')

# Plot histogram for non-churned customers
ax.hist(non_churned_hours, bins=bins, color='blue', alpha=0.5, label='Non-Churned', edgecolor='black')

ax.set_xlabel('Hours Spent on App')
ax.set_ylabel('Frequency')
ax.set_title('Comparison of Hours Spent on App for Churned and Non-Churned Customers')
ax.legend()

# Display the grouped histogram
st.pyplot(fig)

# Title for the dashboard
st.title('Comparison of Customer Satisfaction Score for Churned and Non-Churned Customers')

# Filter out rows with missing values for 'Satisfaction_Score'
df = df.dropna(subset=['SatisfactionScore'])

# Calculate satisfaction scores for churned and non-churned customers
churned_scores = df[df['Churn'] == 1]['SatisfactionScore']
non_churned_scores = df[df['Churn'] == 0]['SatisfactionScore']

# Set up bins for the histogram
bins = range(0, 11, 1)  # Assuming satisfaction score ranges from 0 to 10

# Create a grouped histogram for the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram for churned customers
ax.hist(churned_scores, bins=bins, color='red', alpha=0.5, label='Churned', edgecolor='black')

# Plot histogram for non-churned customers
ax.hist(non_churned_scores, bins=bins, color='blue', alpha=0.5, label='Non-Churned', edgecolor='black')

ax.set_xlabel('Satisfaction Score')
ax.set_ylabel('Frequency')
ax.set_title('Comparison of Customer Satisfaction Score for Churned and Non-Churned Customers')
ax.legend()

# Display the grouped histogram
st.pyplot(fig)

# Title for the dashboard
st.title('Comparison of Marital Status between Churned and Non-Churned Customers')

# Calculate counts of marital status for churned and non-churned customers
churned_marital_status = df[df['Churn'] == 1]['MaritalStatus'].value_counts()
non_churned_marital_status = df[df['Churn'] == 0]['MaritalStatus'].value_counts()

# Get unique marital status values
unique_marital_status = df['MaritalStatus'].unique()

# Create a stacked bar chart for the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for churned customers
ax.bar(unique_marital_status, churned_marital_status, color='red', label='Churned')

# Plot bars for non-churned customers on top of churned bars
ax.bar(unique_marital_status, non_churned_marital_status, bottom=churned_marital_status, color='blue', label='Non-Churned')

ax.set_xlabel('Marital Status')
ax.set_ylabel('Count')
ax.set_title('Comparison of Marital Status between Churned and Non-Churned Customers')
ax.legend()

# Display the stacked bar chart
st.pyplot(fig)

# Title for the dashboard
st.title('Comparison of Customer Complaints between Churned and Non-Churned Customers')

# Calculate counts of complaints for churned and non-churned customers
churned_complaints = df[df['Churn'] == 1]['Complain'].value_counts()
non_churned_complaints = df[df['Churn'] == 0]['Complain'].value_counts()

# Get unique complaint categories
unique_complaints = df['Complain'].unique()

# Create a grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for churned customers
ax.bar(unique_complaints, churned_complaints, color='red', label='Churned')

# Plot bars for non-churned customers with a slight offset
ax.bar([x + 0.4 for x in unique_complaints], non_churned_complaints, color='blue', label='Non-Churned')

ax.set_xlabel('Complaint')
ax.set_ylabel('Count')
ax.set_title('Comparison of Customer Complaints between Churned and Non-Churned Customers')
ax.legend()

# Display the grouped bar chart
st.pyplot(fig)

# Title for the dashboard
st.title('Comparison of Number of Orders between Churned and Non-Churned Customers')

# Calculate total number of orders for churned and non-churned customers
churned_orders = df[df['Churn'] == 1]['OrderCount'].sum()
non_churned_orders = df[df['Churn'] == 0]['OrderCount'].sum()

# Create a grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars for churned and non-churned customers
ax.bar(['Churned', 'Non-Churned'], [churned_orders, non_churned_orders], color=['red', 'blue'])

ax.set_xlabel('Customer Churn Status')
ax.set_ylabel('Total Orders')
ax.set_title('Comparison of Number of Orders between Churned and Non-Churned Customers')

# Display the grouped bar chart
st.pyplot(fig)


# Title for the dashboard
st.title('Comparison of Coupon Usage between Churned and Non-Churned Customers')

# Calculate coupon usage for churned and non-churned customers
churned_coupon_usage = df[df['Churn'] == 1]['CouponUsed'].value_counts()
non_churned_coupon_usage = df[df['Churn'] == 0]['CouponUsed'].value_counts()

# Create a grouped bar chart for the comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Plot bars for churned customers
ax.bar(churned_coupon_usage.index, churned_coupon_usage.values, color='red', label='Churned')

# Plot bars for non-churned customers with a slight offset
ax.bar([x + 0.4 for x in non_churned_coupon_usage.index], non_churned_coupon_usage.values, color='blue', label='Non-Churned')

ax.set_xlabel('Coupon Used')
ax.set_ylabel('Count')
ax.set_title('Comparison of Coupon Usage between Churned and Non-Churned Customers')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not Used', 'Used'])
ax.legend()

# Display the grouped bar chart
st.pyplot(fig)


# Title for the dashboard
st.title('Comparison of Days Since Last Order between Churned and Non-Churned Customers')

# Filter out rows with negative values for 'Days_Since_Last_Order'
df = df[df['DaySinceLastOrder'] >= 0]

# Calculate days since last order for churned and non-churned customers
churned_days_since_last_order = df[df['Churn'] == 1]['DaySinceLastOrder']
non_churned_days_since_last_order = df[df['Churn'] == 0]['DaySinceLastOrder']

# Set up bins for the histogram
bins = range(0, int(max(df['DaySinceLastOrder'])) + 1, 5)  # Adjust bin size as needed

# Create a grouped histogram for the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram for churned customers
ax.hist(churned_days_since_last_order, bins=bins, color='red', alpha=0.5, label='Churned', edgecolor='black')

# Plot histogram for non-churned customers
ax.hist(non_churned_days_since_last_order, bins=bins, color='blue', alpha=0.5, label='Non-Churned', edgecolor='black')

ax.set_xlabel('Days Since Last Order')
ax.set_ylabel('Frequency')
ax.set_title('Comparison of Days Since Last Order between Churned and Non-Churned Customers')
ax.legend()

# Display the grouped histogram
st.pyplot(fig)

# Drop non-numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Title for the dashboard
st.title('Correlation Matrix for Customer Dataset')

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Plot correlation matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')

# Display the correlation matrix heatmap
st.pyplot(plt)



# Descriptive statistics
st.subheader('Descriptive Statistics:')
st.write(df.describe())

st.set_option('deprecation.showPyplotGlobalUse', False)

# Histograms
st.subheader('Histograms:')
for column in df.columns:
    if df[column].dtype == 'float64':
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')
        st.pyplot()

# Box plots
st.subheader('Box Plots:')
for column in df.columns:
    if df[column].dtype == 'float64':
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        st.pyplot()


# Check for missing values
missing_values = df.isnull().sum()

# Display missing values
st.write("Missing Values:")
st.write(missing_values)

# Replace missing values with median
for col in df.columns:
    if df[col].dtype != object:  # Exclude non-numeric columns
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)


#Logistic regression model

# Prepare features (X) and target variable (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Linear regressiopn model

# Prepare features (X) and target variable (y)
X = df.drop('OrderCount', axis=1)  # Features
y = df['OrderCount']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)


#Random RandomForestClassifier
# Prepare features (X) and target variable (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#Decision tree Model
# Prepare features (X) and target variable (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree (optional)
plt.figure(figsize=(12, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=['0', '1'], filled=True)
st.pyplot(fig)

#Principal Component Analysis
# Prepare features (X) and target variable (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Encode categorical variables (if necessary)
X = pd.get_dummies(X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=10)  # Choose the number of principal components
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a classifier (e.g., Random Forest) using the PCA-transformed features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


#line charts

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["Churn", "OrderCount", "DaySinceLastOrder"])

st.line_chart(chart_data)
















# Add a section for data source (optional)
st.sidebar.markdown('Data source: [Ecommerce Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)')






