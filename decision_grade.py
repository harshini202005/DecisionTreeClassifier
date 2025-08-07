import pandas as pd

df=pd.read_csv('grade.csv')
df

df.info()

df.head()

df.isnull().sum()

if 'High School GPA' in df.columns:
  df['High School GPA'].fillna(df['High School GPA'].mean(),inplace=True)
if 'ExtraCuriccular' in df.columns:
  df['ExtraCuriccular'].fillna(df['ExtraCuriccular'].mode()[0],inplace=True)
if 'Hours studied' in df.columns:
  df['Hours studied'].fillna(df['Hours studied'].median(),inplace=True)
print("\nMissing values after handling:\n",df.isnull().sum())

import numpy as np
Q1 = df['ExtraCuriccular'].quantile(0.25)
Q3 = df['ExtraCuriccular'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['ExtraCuriccular'] = np.where(df['ExtraCuriccular'] > upper_bound, upper_bound,
                          np.where(df['ExtraCuriccular'] < lower_bound, lower_bound, df['ExtraCuriccular']))

print("Modified ExtraCurricular column:\n", df)

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
df['Entrance Exam']= label_encoder.fit_transform(df['Entrance Exam'])

df['Entrance Exam'].unique()

df['Internet availability']=label_encoder.fit_transform(df['Internet availability'])
df['Internet availability'].unique()

df

X = df.drop(['Parental income', 'Distance from college(km)','Result'],axis=1)
features=['High School GPA','ExtraCuriccular','Hours studied','Entrance Exam','Internet availability']
X=df[features]

X

y=df['Result']
y

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

input_data = pd.DataFrame([[7.58, 5, 3, 5, 2]], columns=['High School GPA','ExtraCuriccular',
       'Hours studied','Entrance Exam','Internet availability'])
prediction =model.predict(input_data)
print("The result according to the given data is",prediction)

import matplotlib.pyplot as plt
gpa=df['High School GPA']
hours=df['Hours studied']
plt.scatter(gpa,hours, color='purple')
plt.title('GPA vs Hours Studied')
plt.xlabel('High School GPA')
plt.ylabel('Hours Studied')
plt.show()

from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy of the model:", accuracy)

import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
