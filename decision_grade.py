import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pickle


df = pd.read_csv("grade.csv")

X = df.drop("Result", axis=1)
y = df["Result"]


le_exam = LabelEncoder()
le_internet = LabelEncoder()
le_result = LabelEncoder()

X["Entrance Exam"] = le_exam.fit_transform(X["Entrance Exam"])
X["Internet availability"] = le_internet.fit_transform(X["Internet availability"])
y = le_result.fit_transform(y)

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


pickle.dump(model, open("model.pkl", "wb"))
