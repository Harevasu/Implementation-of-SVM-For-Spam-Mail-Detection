# EXPERIMENT NO: 11
## Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program & Output:
### Developed by: HAREVASU S
### RegisterNumber: 212223230069

### Program to implement the SVM For Spam Mail Detection..
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
```
```c
data.head()
```
![image](https://github.com/user-attachments/assets/90f1d39a-8d75-4677-b226-8852bcf2bbfb)
```c
data.info()
```
![image](https://github.com/user-attachments/assets/07754f65-3117-4af1-85c0-4cbf659d39b5)

```c
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/2e759493-1bda-4a5c-b94b-f4c71669a9cb)

```c
x=data['v2'].values
y=data['v1'].values
y.shape
```
![image](https://github.com/user-attachments/assets/4cd4e197-a2e6-47fb-ac95-d7972a5ef6ca)

```c
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```
![image](https://github.com/user-attachments/assets/911eac86-7409-45cb-be45-24813b76ec8d)
```c
x_test.shape
```
![image](https://github.com/user-attachments/assets/be746ec6-9b30-4ed6-9b20-b6fa4e84cb0a)
```c
y_train.shape
```
![image](https://github.com/user-attachments/assets/74300b3f-c807-469d-ad84-ac2aaede1728)
```c
y_test.shape
```
![image](https://github.com/user-attachments/assets/c3f41c93-c3f1-459a-b6ee-4b25d2dad9a2)
```c
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train.shape
```
![image](https://github.com/user-attachments/assets/8b8e50a8-bc00-43b3-891d-bc39d94757e2)
```c
x_test.shape
```
![image](https://github.com/user-attachments/assets/6cf50893-0a0a-4880-8a57-c2d64418fc0b)
```c
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/1959e246-1d8b-42f8-a2eb-46220b6627ba)
```c
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/7ecb36da-8693-404a-ac46-cb518d6130d3)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
