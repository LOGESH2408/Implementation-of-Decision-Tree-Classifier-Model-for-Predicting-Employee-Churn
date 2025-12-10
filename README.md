# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: LOGESHWARAN S

RegisterNumber:  25007255
*/
```
import pandas as pd
from IPython.display import display

data = pd.read_csv("Employee.csv")


print("===== DATA HEAD (5 ROWS) =====")
display(data.head())

print("\n===== DATA INFO =====")
print(data.info())


print("\n===== MISSING VALUES =====")
display(data.isnull().sum().to_frame("Missing Values"))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

print("\n===== X HEAD (5 ROWS) =====")
display(x.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100
)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)

print("\n===== MODEL ACCURACY =====")
print("Accuracy:", accuracy)
print("\n===== PREDICTION =====")
print(dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]]))


## Output:
<img width="1242" height="258" alt="Screenshot 2025-12-10 091149" src="https://github.com/user-attachments/assets/d3287d81-4c06-419a-b705-accdae449768" />

<img width="642" height="422" alt="Screenshot 2025-12-10 091344" src="https://github.com/user-attachments/assets/4796a505-bd84-430d-b4fc-3eefeae7a1f6" />

<img width="355" height="428" alt="Screenshot 2025-12-10 091356" src="https://github.com/user-attachments/assets/9628df58-a512-4896-83ea-786c2267b4ed" />


<img width="1172" height="280" alt="Screenshot 2025-12-10 091433" src="https://github.com/user-attachments/assets/785250eb-99af-4ba4-bfef-6bb589d3937d" />


<img width="429" height="73" alt="Screenshot 2025-12-10 091450" src="https://github.com/user-attachments/assets/a0e5134c-5da0-46d3-9ecb-d79adf9ee8b3" />


<img width="319" height="124" alt="Screenshot 2025-12-10 091459" src="https://github.com/user-attachments/assets/96867abb-02cd-4161-8c16-9becfddf4527" />







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
