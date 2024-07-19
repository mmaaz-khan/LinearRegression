# Importing numpy and pandas
import pandas as pd
import numpy as np

run = True

# Reading csv file into pandas DataFrame
mydata = pd.read_csv('CO2Data.csv')
print(mydata)
#print(mydata.to_string())

# Manipulating DataFrame into x and y dataframes using the csv column headers
x = mydata[['b0', 'Weight', 'Volume']]
y = mydata[['CO2']]

# Using the .to_numpy() function to change the indivdual x and y dataframes into numpy arrays which act as matrices
x_mat = x.to_numpy()
y_mat = y.to_numpy()

#print(x_mat)
#print(y_mat)

# To find coefficients (bn) of multiple regression you need to do inv(xT * x) * (xT * y)
# Numpy has transpose, inverse and mutliplication methods built in
x_transpose = x_mat.transpose()
xTx = (x_transpose).dot(x_mat)
xTx_inv = np.linalg.inv(xTx)
xTy = (x_transpose).dot(y_mat)
b = xTx_inv.dot(xTy)

# The matrix of b values from b0 to bn is printed, with b0 being the y-intercept
#print(b)

# Finding R-Squared using the formula r^2 = (sum(ypred - ybar)^2)/(sum(y-ybar)^2)
def find_r_squared():
  predvals = []
  for i in range(len(y)):
    ypredval = b[0] + b[1].dot(x_mat[i][1]) + b[2].dot(x_mat[i][2])
    predvals.append(ypredval)
  
  meany = y.mean()
  r_num_T = 0
  r_den_T = 0
  for i in range(len(predvals)):
    r_num = ((predvals[i]) - meany)**2
    r_num_T += r_num
  
    r_den = ((y_mat[i]) - meany)**2
    r_den_T += r_den
  
  r_squared = 1 - (r_num_T/r_den_T)
  
  print(r_squared)
#print(predvals)

# Creating a procedure which takes in the weight and volumes for which you want to predict CO2 emmissions, and creates a formula using the appropriate b values for the variables and y-intercept
def predy(volume, weight):
  predy = b[0] + (weight * b[1]) + (volume * b[2])
  print(predy)

print("----Your data has been loaded. To find the r-squared value, select 1. If you want to predict a value for your linear regression model, select 2. If you want to quit, press 9----")
while run == True:
  selectoption = int(input("\nSelect whether you want 1 or 2 or 9: "))
  while selectoption != 1 and selectoption!= 2 and selectoption != 9:
    print("Not a valid input")
    selectoption = int(input("Select whether you want 1 or 2 or F: "))
  if selectoption == 9:
    print("You have chosen to exit. Leaving system")
    run = False
  elif selectoption == 1:
    find_r_squared()
  else:
    print("Enter the 2 values you want to predict the outcome for (enter in the order of the file)")
    var1entry = float(input("Enter your var1 input: "))
    var2entry = float(input("Enter your var2 input: "))
    predy(var1entry, var2entry)

# Using scikitlearn to check whether my code outputs the correct results
#import pandas
#from sklearn import linear_model

#df = pandas.read_csv("CO2Data.csv")

#X = df[['Weight', 'Volume']]
#y = df['CO2']

#regr = linear_model.LinearRegression()
#regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
#predictedCO2 = regr.predict([[2300, 1300]])

#print(predictedCO2)