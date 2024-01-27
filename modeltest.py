from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
data =pd.read_csv("dataset/data.csv")
X = data.iloc[:, 0:42].values
y = data.iloc[:, 42].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("开始训练")
# 训练SVM模型
clf = svm.SVC(kernel='rbf',C=10,gamma=0.01,)
clf.fit(X_train_scaled, y_train)

# 预测测试集结果并输出精度
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Accuracy score SVM:", acc)