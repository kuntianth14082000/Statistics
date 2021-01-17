from sklearn import datasets

df=datasets.load_iris()

x=df.data
x1=x[:,1]
x2=x[:,2]
x3=x[:,3]
x4=x[:,0]

import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(x1)
plt.hist(x2)
plt.hist(x3)
plt.hist(x4)

#checking for log normal distribution
import numpy as np
import pandas as pd
data=pd.read_csv(r'D:\sabir\python\Datasets\tips.csv')
y1=data.iloc[:,0]
y2=data.iloc[:,1]
plt.hist(y1)
plt.hist(y2)

# checking data belongs to log normal distribution or not by 
# -taking log and is it belongs to Gaussian distribution or not
y2_pro=np.log(y2)
y1_pro=np.log(y1)
plt.hist(y2_pro)
plt.hist(y1_pro)

from sklearn.preprocessing import StandardScaler
z=data.iloc[:,0:2]
z_log=np.log(z)
ss=StandardScaler()
y2_ss=ss.fit_transform(z)

z_log.mean()
z_log.std()

#--------------------------------------------
#finding cavariance & correlation coefficient 
import numpy as np
import pandas as pd
data=pd.read_csv(r'D:\sabir\python\Datasets\tips.csv')
y1=data.iloc[:,0]
y2=data.iloc[:,1]

data.cov()
data.corr()
np.cov(y1,y2)

#calculating spearmens rank correlation
from scipy.stats.stats import spearmanr
sc=spearmanr(y1,y2)[0]

#or using kendalltau rank correlation
from scipy.stats.stats import kendalltau
kt=kendalltau(y1,y2)


#----------------------------------------------
#outliers
#z-score methohd
import numpy as np
data=[12,15,17,15,16,20,16,17,12,15,12,11,10,7,14,16,15,19,20,108,17,100,14,17,105]
outliers=[]

def detect_outliers(data):
    threshould=2
    mean=np.mean(data)
    std=np.std(data)
    
    for i in data:
        z_score=(i-mean)/std
        if np.abs(z_score) > threshould:
            outliers.append(i)
    return outliers

output=detect_outliers(data)

#IQR method
outlier1=[]
data=sorted(data)
quantile1,quantile3=np.percentile(data,[25,75])
#find IQR
iqr_value=quantile3-quantile1
#finding lower bound and upper bound
lower_bound_value=quantile1-(1.5*iqr_value)
upper_bound_value=quantile3+(1.5*iqr_value)
#outliers output
for i in data:
    if  i > 21.5 or i <9.5:
        outlier1.append(i)
        
#Normalization and standardisation
#1.Normalization
import pandas as pd        
dataset=pd.read_csv(r'D:\sabir\python\Datasets\Fahad ML\Wine Video 49\Video 49\Wine.csv',usecols=[0,1,2])
#dataset.columns=['class','Alcohol','Malic_Acid']
dataset.head()
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
scale.fit_transform(dataset[['Alcohol','Malic_Acid']])

#2.Standardisation
from sklearn.preprocessing import StandardScaler
scaling=StandardScaler()
scaling.fit_transform(dataset[['Alcohol','Malic_Acid']])
