### Predicting Students Marks based on how many hours they study. ###
#Business problem

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
path = r"C:\Users\donga\Downloads\ML\student_info.csv"
df=pd.read_csv(path)

print(df.head())
print(df.tail())
print(df.shape)

#discover and visualize the data
df.info()
df.describe()  #We can see that the maximum study hours is 8.99 yet the student score is 86.99, hence we need to figure out what hours student needs to stuuty in order to get more than 90% marks.

plt.scatter(x=df.study_hours, y=df.student_marks, color='red')
plt.xlabel("Student Study Hours")
plt.ylabel("Student Marks")
plt.title("Scattered Plot of Student Study Hours vs Marks")
plt.show()

#Prepare the data for machine learning algorithm
#data cleaning
df.isnull().sum(0)  #to see the number of null values in table
df.mean()  #to see the mean of each column
df2=df.fillna(df.mean())
df2.isnull().sum()
df2.head()

#split dataset
X = df2.drop("student_marks", axis="columns")  #features
y = df2.drop("study_hours", axis="columns")  #target variable
print("shape of X:", X.shape)
print("shape of Y:", y.shape)

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=51)
print("shape of X_train:", X_train.shape)
print("shape of Y_train:", y_train.shape)
print("shape of X_test:", X_test.shape)
print("shape of Y_test:", y_test.shape)

#SELECT A MODEL AND TRAIN IT
#Linear Regression model suits best for this problem
# y= m*x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.coef_  #to see the coefficients of the model
lr.intercept_  #to see the intercept of the model

m=3.93
c=-50.44
y= m*4+c
y

lr.predict([[4]])[0][0].round(2)  #predicting the marks for a student who studies for 4 hours
y_pred=lr.predict(X_test)
pd.DataFrame(np.c_[X_test, y_test, y_pred], columns=["study_hours","student_marks_original","student_marks_predicted"])  #combining the test data with actual and predicted values   

#Fine tune the model
lr.score(X_test, y_test)  #to see the accuracy of the model
#The accuracy is 0.95 which is very good, hence we can use this model

plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.plot(X_train, lr.predict(X_train), color='blue')

#Present your solution
#save ML model
import joblib
joblib.dump(lr, "student_marks_prediction_model.pkl")  #saving the model to a file

model=joblib.load("student_marks_prediction_model.pkl")  #loading the model from a file
model.predict([[5]])[0][0]

# Launch, Monitor, and Maintain your system
