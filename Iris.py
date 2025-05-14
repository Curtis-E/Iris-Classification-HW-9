import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# https://www.kaggle.com/datasets/uciml/iris
df = pd.read_csv("Iris.csv")

# Drops ID feature
df.drop(columns=['Id'], inplace=True)

y = df['Species'].copy().to_numpy()
X = df.drop(columns=['Species']).to_numpy()

#Split Train and Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Random Forest Classifier
clf = RandomForestClassifier(max_depth=12, oob_score=True, verbose=3, n_jobs=-1)
clf.fit(X_train, y_train)

#Plot features
importances = pd.DataFrame(clf.feature_importances_, index=df.drop(columns=['Species']).columns)
importances.plot.bar()
plt.show()

# Print scores
print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

# Display the confusion matrix
cm = confusion_matrix(y_test, clf.predict(X_test))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
