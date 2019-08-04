from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
# print(X[:2, :])
# print(y)
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
self.model = LogisticRegression(random_state=0, solver='liblinear')
print(clf.predict(X[:1, :]))

print(clf.predict_proba(X[:1, :]))


print(clf.score(X, y))