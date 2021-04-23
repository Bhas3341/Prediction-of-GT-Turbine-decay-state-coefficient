import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model


header_name=pd.read_table('/Users/bhaskaryuvaraj/Downloads/UCI CBM Dataset/Features.txt')
header=['lp','Ship speed','GTT','GTn','GGn','Ts','Tp','T48','T1','T2','P48','P1','P2','Pexh','TIC','Fuel flow',
         'GT Compressor decay state coefficient','GT Turbine decay state coefficient	']	
by=pd.read_table('/Users/bhaskaryuvaraj/Downloads/UCI CBM Dataset/data.txt',header=None, delimiter=r"\s+",names=header)

by.columns															
by.dtypes
by.describe()
#1 - Lever position (lp) [ ]
#2 - Ship speed (v) [knots]
#3 - Gas Turbine shaft torque (GTT) [kN m]
#4 - Gas Turbine rate of revolutions (GTn) [rpm]
#5 - Gas Generator rate of revolutions (GGn) [rpm]
#6 - Starboard Propeller Torque (Ts) [kN]
#7 - Port Propeller Torque (Tp) [kN]
#8 - HP Turbine exit temperature (T48) [C]
#9 - GT Compressor inlet air temperature (T1) [C]
#10 - GT Compressor outlet air temperature (T2) [C]
#11 - HP Turbine exit pressure (P48) [bar]
#12 - GT Compressor inlet air pressure (P1) [bar]
#13 - GT Compressor outlet air pressure (P2) [bar]
#14 - Gas Turbine exhaust gas pressure (Pexh) [bar]
#15 - Turbine Injecton Control (TIC) [%]
#16 - Fuel flow (mf) [kg/s]
#17 - GT Compressor decay state coefficient.
#18 - GT Turbine decay state coefficient. 
#----------------------------------Lets do EDA-----------------------------------------
by.plot(kind='scatter',x='lp',y='GT Compressor decay state coefficient')
#----------------------------------end of EDA-----------------------------------------

by.isnull().sum()
#no missing values

#finding outliers
plt.boxplot(by['lp'])
plt.boxplot(by['Ship speed'])
plt.boxplot(by['GTT'])
plt.boxplot(by['GTn'])
plt.boxplot(by['GGn'])
plt.boxplot(by['Ts'])
plt.boxplot(by['Tp'])
plt.boxplot(by['T48'])
plt.boxplot(by['T1'])
plt.boxplot(by['T2'])
plt.boxplot(by['P48'])
plt.boxplot(by['P1'])
plt.boxplot(by['P2'])
plt.boxplot(by['Pexh'])
plt.boxplot(by['TIC'])  #has outlier
plt.boxplot(by['Fuel flow'])
plt.boxplot(by['GT Compressor decay state coefficient'])
plt.boxplot(by['GT Turbine decay state coefficient\t'])

#to remove the outlier
def remove_outlier(d,c):
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    iqr=q3-q1
    ub=q3+1.53*iqr
    lb=q1-1.53*iqr
    result=d[(d[c]>lb) & (d[c]<ub)]
    return result

by=remove_outlier(by,'TIC')
plt.boxplot(by['TIC']) 

#now to find the corelation btw the columns
#correlated_features=set()
#correlated_matrix=by.drop('GT Compressor decay state coefficient',axis=1).corr()
#
#for i in range(len(correlated_matrix.columns)):
#    for j in range(i):
#        if abs(correlated_matrix.iloc[i,j])>0.8:
#            colname=correlated_matrix.columns[i]
#            correlated_features.add(colname)
#            
#print(correlated_features)
y=by['GT Compressor decay state coefficient'].copy()
x=by.drop('GT Compressor decay state coefficient',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
x_train.columns

#linear regression
lm=linear_model.LinearRegression()
project=lm.fit(x_train,y_train)
#training accuracy
print(project.score(x_train,y_train))
#0.9009048016175766
predicted_y=lm.predict(x_test)
#predicted accuracy
print(project.score(x_test,y_test))
#0.8988897457439172