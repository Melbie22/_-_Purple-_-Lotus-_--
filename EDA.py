import pandas as pd

df = pd.read_csv('diabetes.csv')
df_2 = df.loc[(df['Glucose']!=0) & (df['BloodPressure']!=0) & (df['SkinThickness']!=0) & (df['Insulin']!=0) & (df['BMI']!=0)]

df['Glucose'].replace(0, df_2['Glucose'].mean(), inplace=True)
df['BloodPressure'].replace(0, df_2['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].replace(0, df_2['SkinThickness'].mean(), inplace=True)
df['Insulin'].replace(0, df_2['Insulin'].mean(), inplace=True)
df['BMI'].replace(0, df_2['BMI'].mean(), inplace=True)
df.drop_duplicates(inplace=True)

df.to_csv('diabetes_cleaned.csv', index=False)