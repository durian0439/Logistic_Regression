import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 나이와 연봉으로 분석해서, 물건을 구매할지 안할지를 분류하기.

df= pd.read_csv('Social_Network_Ads.csv')

df.head() #컬럼과 데이터의 앞부분 확인

df.isna().sum() #Nan data 확인

X = df.iloc[:,2:3+1] # Age/ EstimatedSalary 컬럼을 X로 설정

y =df['Purchased'] 

X.head()

y.head()

#  피쳐 스케일링

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

X


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size = 0.2, random_state = 0)

X.shape #차원확인

X_train.shape

X_test.shape

X_test



from sklearn.linear_model import LogisticRegression #Logistic_Regression 학습

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred

y_test.values



#36세 38,000달러의 조건에서 구매여부를 예측하기

new = np.array([36,38000]).reshape(1,-1)

new

# 피처 스케일링을 위에서 학습한 것을 사용해야 함. 

new1 = sc.transform(new)

# 예측하기

y_pred1 = classifier.predict(new1)

y_pred1


# Confusion Matrix를 통해 정확도 / 정밀도 / 적중률 확인하기

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

(cm[0][0]+cm[1][1])/cm.sum()          # 정확도

cm[1][1]/(cm[1][0]+cm[1][1])          # 정밀도

cm[1][1]/cm[1][0]+cm[1][1])           # 적중률

