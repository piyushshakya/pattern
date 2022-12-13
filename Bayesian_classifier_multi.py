from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

print(gnb.score(X_test, y_test))

plot_confusion_matrix(gnb, X_test, y_test)

y_pred = gnb.predict(X_test)
print(classification_report(y_test, y_pred))
