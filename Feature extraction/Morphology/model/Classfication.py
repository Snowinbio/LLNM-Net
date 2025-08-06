import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
import pickle

class LogisticRegressionWithANOVA:
    def __init__(self, result_df, feature_cols, label_col, logistic_train_bool):
        self.df = result_df
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.anova_scores = None
        self.p_values = None
        self.model = None
        self.predictions = None
        self.logistic_train_bool = logistic_train_bool

    def read_data_and_preprocess(self):
        X = self.df[self.feature_cols]
        y = self.df[self.label_col]

        '''
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.X = X_scaled
        '''

        self.X = X.values
        self.y = y.values.ravel()

    def train_model_and_anova_feature_selection(self):
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            fit_intercept=True,
            class_weight='balanced',
            max_iter=100,
            random_state=42,
            warm_start=False
        )
        self.model.fit(self.X, self.y)
        if self.logistic_train_bool:
            with open('./weights/Logistic.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        else:
            with open('./weights/Logistic.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
        self.anova_scores, self.p_values = f_classif(self.X, self.y)

    def print_feature_importance(self):
        feature_importance = pd.DataFrame(
            {'Feature': self.feature_cols, 'ANOVA Score': self.anova_scores, 'P-Value': self.p_values})
        print("ANOVA Feature Importance:\n", feature_importance)

    def predict_and_evaluate(self):
        self.predictions = self.model.predict(self.X)
        accuracy = accuracy_score(self.y, self.predictions)
        precision = precision_score(self.y, self.predictions)
        auc = roc_auc_score(self.y, self.model.predict_proba(self.X)[:, 1])
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test AUC: {auc:.4f}")

    def run(self):
        self.read_data_and_preprocess()
        self.train_model_and_anova_feature_selection()
        self.print_feature_importance()
        self.predict_and_evaluate()

        # print(self.X)

