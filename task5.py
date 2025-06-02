import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\30 day intesnhip online\heart.csv")  
print("Dataset shape:", df.shape)
print(df.head())

print(df.isnull().sum())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
print("\nDecision Tree Accuracy on test set:", clf.score(X_test, y_test))

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.title('Decision Tree')
plt.show()

depths = range(1, 11)
accuracies = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies.append(acc)

plt.figure(figsize=(8,5))
plt.plot(depths, accuracies, marker='o', color='blue')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Impact of Tree Depth on Accuracy')
plt.grid()
plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("\nRandom Forest Accuracy on test set:", rf.score(X_test, y_test))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], color='green', align='center')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

scores = cross_val_score(rf, X, y, cv=5)
print("\nCross-validation scores:", scores)
print("Average CV accuracy:", scores.mean())

