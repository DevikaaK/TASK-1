import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess_titanic_data(df, visualize=True):
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else np.nan)
    deck_impute_map = {1: 'C', 2: 'E', 3: 'G'}
    df['Deck'] = df['Deck'].fillna(df['Pclass'].map(deck_impute_map))
    df = df.drop('Cabin', axis=1)

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.dropna()

    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df['Age'])
        plt.title("Age - Before Outlier Removal")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df['Fare'])
        plt.title("Fare - Before Outlier Removal")
        plt.tight_layout()
        plt.show()

    Q1_age = df['Age'].quantile(0.25)
    Q3_age = df['Age'].quantile(0.75)
    IQR_age = Q3_age - Q1_age
    df = df[(df['Age'] >= Q1_age - 1.5 * IQR_age) & (df['Age'] <= Q3_age + 1.5 * IQR_age)]

    Q1_fare = df['Fare'].quantile(0.25)
    Q3_fare = df['Fare'].quantile(0.75)
    IQR_fare = Q3_fare - Q1_fare
    df = df[(df['Fare'] >= Q1_fare - 1.5 * IQR_fare) & (df['Fare'] <= Q3_fare + 1.5 * IQR_fare)]

    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df['Age'])
        plt.title("Age - After Outlier Removal")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df['Fare'])
        plt.title("Fare - After Outlier Removal")
        plt.tight_layout()
        plt.show()

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['Embarked', 'Deck'], drop_first=True)

    scaler = MinMaxScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    return df

if __name__ == "__main__":
    df = pd.read_csv('titanic.csv')

    print("Step 1: Dataset Overview (First 5 Rows):")
    print(df.head())

    print("\nData Types & Missing Values:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values Count:")
    print(df.isnull().sum())

    cleaned_df = preprocess_titanic_data(df)

    print("\nCleaned Dataset Preview (First 5 Rows):")
    print(cleaned_df.head())

    print("\nCleaned Dataset Shape:", cleaned_df.shape)
