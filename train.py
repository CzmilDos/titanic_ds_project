import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class Train:
    def load_data(self, file_path):
        """Load data from a CSV file."""
        return pd.read_csv(file_path, sep=',')

    def preprocess_data(self, data):
        """Preprocess data and return features (X) and labels (y)."""
        X = data[['Pclass', 'Sex', 'Age']]
        X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
        y = data['Survived']
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def scale_data(self, X_train, X_test):
        """Scale data using StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_model(self, model, X_train, y_train):
        """Train the specified model."""
        model.fit(X_train, y_train)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model and return metrics."""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        return accuracy, report, precision, recall, f1, confusion

    def main(self):
        # Charger les données
        train_data = self.load_data('data/train.csv')

        # Prétraiter les données
        X, y = self.preprocess_data(train_data)

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Normaliser les données si nécessaire
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)

        # Entraîner des modèles
        model_lr = LogisticRegression()
        model_dt = DecisionTreeClassifier()
        model_rf = RandomForestClassifier()

        self.train_model(model_lr, X_train_scaled, y_train)
        self.train_model(model_dt, X_train_scaled, y_train)
        self.train_model(model_rf, X_train_scaled, y_train)

        # Évaluer les modèles
        results_lr = self.evaluate_model(model_lr, X_test_scaled, y_test)
        results_dt = self.evaluate_model(model_dt, X_test_scaled, y_test)
        results_rf = self.evaluate_model(model_rf, X_test_scaled, y_test)

        # Afficher les résultats (à adapter selon vos besoins)
        print("Logistic Regression Results:")
        print("Accuracy:", results_lr[0])
        print("Classification Report:\n", results_lr[1])
        print("Precision:", results_lr[2])
        print("Recall:", results_lr[3])
        print("F1-score:", results_lr[4])
        print("Confusion Matrix:\n", results_lr[5])

        # Répéter pour les autres modèles...


if __name__ == "__main__":
    model_instance = Train()
    model_instance.main()
