# Importing libraries
import pandas as pd
import joblib 
from sklearn.ensemble import RandomForestClassifier

data = ('diabetes_cleaned.csv')
df = pd.read_csv(data)
# df = df.drop(df.columns[:2], axis=1)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Saving the model
joblib.dump(model, 'diabetes_model.pkl')