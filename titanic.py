import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.tail()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women=sum(women)/len(women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men=sum(men)/len(men)
print("% of women who survived:",rate_women)
print("% of men who survived : ", rate_men)
from sklearn.ensemble import RandomForestClassifier
target= train_data["Survived"]
target
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, target)
predictions = model.predict(X_test)
predictions
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
output.head()
