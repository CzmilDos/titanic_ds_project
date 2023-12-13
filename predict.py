import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Preprocess data and return features (X)."""
    X = data[['Pclass', 'Sex', 'Age']]
    X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
    return X


def main():
    # Charger le modèle entraîné du fichier train.py
    model = joblib.load('trained_model.pkl')

    # Charger les données de test
    test_data = load_data('data/test.csv')
    
    # Prétraiter les données de test
    X_test = preprocess_data(test_data)
    
    # Charger le scaler sauvegardé dans train.py
    scaler = joblib.load('scaler.pkl')

    # Appliquer le scaler sur les données de test
    X_test_scaled = scaler.transform(X_test)
    
    # Effectuer la prédiction
    predictions = model.predict(X_test_scaled)

    # Ajouter les prédictions à un DataFrame
    results = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': predictions
    })

    # Enregistrer les résultats dans un fichier CSV
    results.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'.")


if __name__ == '__main__':
    main()
