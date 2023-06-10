# task1
GRIP : THE SPARKS FOUNDATION
DATA SCIENCE AND BUSINESS ANALYTIC INTERN

TASK 1 : PREDICTION USING SUPERVISED ML

AUTHOR : MOHIT SHARMA

# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
​
# Reading data from remote link
data = "http://bit.ly/w-data"
ds= pd.read_csv(data)
ds
​
​
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
10	7.7	85
11	5.9	62
12	4.5	41
13	3.3	42
14	1.1	17
15	8.9	95
16	2.5	30
17	1.9	24
18	6.1	67
19	7.4	69
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
ds.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hours   25 non-null     float64
 1   Scores  25 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 528.0 bytes
Separating Dependent and Independent Variables
x=ds.iloc[:,0:1].values
x
array([[2.5],
       [5.1],
       [3.2],
       [8.5],
       [3.5],
       [1.5],
       [9.2],
       [5.5],
       [8.3],
       [2.7],
       [7.7],
       [5.9],
       [4.5],
       [3.3],
       [1.1],
       [8.9],
       [2.5],
       [1.9],
       [6.1],
       [7.4],
       [2.7],
       [4.8],
       [3.8],
       [6.9],
       [7.8]])
y=ds.iloc[:,1:2].values
y
array([[21],
       [47],
       [27],
       [75],
       [30],
       [20],
       [88],
       [60],
       [81],
       [25],
       [85],
       [62],
       [41],
       [42],
       [17],
       [95],
       [30],
       [24],
       [67],
       [69],
       [30],
       [54],
       [35],
       [76],
       [86]], dtype=int64)
Scatter plot between Dependent and Independent Variables
# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

Separating train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
plt.scatter(x_train,y_train)
<matplotlib.collections.PathCollection at 0x201c9835720>

Linear Regression
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x_train,y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Predicting the scores of the students
y_predict=linear_reg.predict(x_test)
y_predict
array([[16.88414476],
       [33.73226078],
       [75.357018  ],
       [26.79480124],
       [60.49103328]])
Predicting avtual scores
y_test
array([[20],
       [27],
       [69],
       [30],
       [62]], dtype=int64)
Checking the accuracy of the Model
plt.scatter(x_train,y_train)
plt.plot(x_test,y_predict,color="red")
[<matplotlib.lines.Line2D at 0x201ca2465c0>]

Testing data - in hours
linear_reg.predict([[9.25]])
array([[93.69173249]])
Evaluating the Model
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
0.9454906892105356
