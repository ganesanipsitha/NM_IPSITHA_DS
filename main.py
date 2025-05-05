# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')

# Data Collection
def load_data(file_path):
    """Load dataset from CSV file"""
    return pd.read_csv(file_path)

# Data Cleaning
def clean_data(df):
    """Handle missing values, duplicates, and inconsistencies"""
    # Drop columns with too many missing values (more than 50%)
    df = df.dropna(thresh=len(df)/2, axis=1)
    
    # Fill numerical missing values with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

# Feature Engineering
def engineer_features(df):
    """Create new features and apply transformations"""
    # Example: Create age of house feature if YearBuilt exists
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = pd.Timestamp.now().year - df['YearBuilt']  # FIXED HERE
    
    # Example: Create total area feature
    if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Apply log transformation to skewed numerical features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    skewed_cols = df[num_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    skewness_threshold = 0.75
    skewed_cols = skewed_cols[abs(skewed_cols) > skewness_threshold]
    
    for col in skewed_cols.index:
        if col != 'SalePrice':  # Skip target variable
            if df[col].min() > 0:  # Log transform only works for positive values
                df[col] = np.log1p(df[col])
    
    return df

# Exploratory Data Analysis
def perform_eda(df, target_col='SalePrice'):
    """Generate visualizations to understand data patterns"""
    plt.figure(figsize=(15, 10))
    
    # Target variable distribution
    plt.subplot(2, 2, 1)
    sns.histplot(df[target_col], kde=True)
    plt.title('Target Variable Distribution')
    
    # Correlation heatmap
    plt.subplot(2, 2, 2)
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    sns.heatmap(corr[abs(corr[target_col]) > 0.5], annot=True, cmap='coolwarm')
    plt.title('Feature Correlation with Target')
    
    # Pair plot for top correlated features
    plt.subplot(2, 1, 2)
    top_features = corr[target_col].sort_values(ascending=False).index[1:6]
    sns.pairplot(df[top_features].join(df[target_col]))
    plt.suptitle('Pair Plot of Top Correlated Features', y=1.02)
    
    plt.tight_layout()
    plt.show()

# Model Building and Evaluation
def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple regression models"""
    # Define preprocessing for numerical and categorical features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selector', SelectKBest(score_func=f_regression, k='all')),
            ('regressor', model)])
        
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV R2 Mean': cv_scores.mean(),
            'CV R2 Std': cv_scores.std()
        }
        
        # Plot predictions vs actual
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'{name} Predictions vs Actual')
        plt.show()
    
    return pd.DataFrame(results).T

# Main function
def main():
    # Load data (replace with your actual file path)
    try:
        file_path = 'train.csv'  # Update this path
        df = load_data(file_path)
    except FileNotFoundError:
        print("Please download the dataset from Kaggle and update the file path.")
        return
    
    # Data cleaning
    df = clean_data(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    # EDA
    perform_eda(df)
    
    # Prepare data for modeling
    target_col = 'SalePrice'
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset.")
        return
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Apply log transform to target if skewed
    if abs(y.skew()) > 0.75:
        y = np.log1p(y)
        print("Applied log transformation to target variable due to skewness.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Build and evaluate models
    results = build_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\nModel Evaluation Results:")
    print(results.sort_values(by='R2', ascending=False))
    
    # Feature importance for the best model
    best_model_name = results['R2'].idxmax()
    print(f"\nBest model: {best_model_name}")

if __name__ == "__main__":
    main()