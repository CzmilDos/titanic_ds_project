import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Preprocess data and return features (X) and labels (y)."""
    X = data[['Pclass', 'Sex', 'Age']]
    X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
    y = data['Survived']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    """Scale data using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and return metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, confusion


def main():
    # Charger les données
    train_data = load_data('data/train.csv')

    # Prétraiter les données
    X, y = preprocess_data(train_data)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Normaliser les données si nécessaire
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Entraîner le modèle
    model = train_model(X_train_scaled, y_train)

    # Évaluer le modèle
    results = evaluate_model(model, X_test_scaled, y_test)

    # Afficher les résultats
    print("Model Evaluation Results:")
    print("Accuracy:", results[0])
    print("Precision:", results[1])
    print("Recall:", results[2])
    print("F1-score:", results[3])
    print("Confusion Matrix:\n", results[4])

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-validation Accuracy Scores:", cv_scores)


if __name__ == "__main__":
    main()
