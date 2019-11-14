from datetime import datetime
from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score


class RandomForest(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def _calculate_chi2(self):
        chi2_selector = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=10)
        X = self.dataframe.drop(['tuid', 'sex'], axis=1).astype(int)
        y = self.dataframe['sex']
        X_kbest = chi2_selector.fit_transform(X, y)
        print('Reduced number of features:', X_kbest.shape[1])
        return X_kbest

    def run_model(self):
        X = self._calculate_chi2()
        y = self.dataframe['sex']
        dt = datetime.now()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=dt.second)
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))