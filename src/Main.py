import matplotlib.pyplot as plt
from ExtractSaveFeature import ExtractSaveFeature

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


# =========================================================
# Feature Extracted
# =========================================================

trainDataGlobal, trainLabelsGlobal = ExtractSaveFeature.get_train("resource/train_labels.txt")

testDataGlobal, testLabelsGlobal = ExtractSaveFeature.get_test("resource/test_labels.txt")

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))






# =========================================================
# Model Training
# =========================================================

# no.of.trees for Random Forests
num_trees = 100

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9, kernel='rbf')))

for name, model in models:

    model.fit(trainDataGlobal, trainLabelsGlobal)

    predict_y = model.predict(testDataGlobal)

    print("==========", name, "=============")
    c_matrix = confusion_matrix(testLabelsGlobal, predict_y)
    print(name, "AccuracyScore :", accuracy_score(testLabelsGlobal, predict_y))


plt.show()