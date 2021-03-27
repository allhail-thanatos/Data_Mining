import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.compose import ColumnTransformer


dataset=pd.read_csv("UMP.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(dataset)
print(x)
print(y)

print('\n==================menghilangkan nan==============\n')
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:, 2:3])
x[:, 2:3] = imputer.transform(x[:, 2:3])
print(x)

print('\n=================encoding data kategori(atribut)================\n')
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)


print('\n=================encoding data kategori(class/label)=========================\n')
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

print('\n====================Splitting dataset=================\n')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)


sc = StandardScaler()
x_train[:, 2:4] = sc.fit_transform(x_train[:, 2:3])
x_test[:, 2:4] = sc.transform(x_test[:, 2:3])

print('\n=================FEATURE SCALING================\n')
print(x_train)
print(x_test)