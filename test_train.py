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
import pytest

from train import load_data, preprocess_data, split_data, scale_data, train_model, evaluate_model

@pytest.fixture
def sample_data():
    # Créer un DataFrame fictif pour les tests
    data = pd.DataFrame({
        'Pclass': [1, 2, 3, 1, 2],
        'Sex': ['male', 'female', 'male', 'female', 'male'],
        'Age': [22, 30, 25, 35, 40],
        'Survived': [1, 0, 1, 0, 1]
    })
    return data

def test_load_data(sample_data):
    # Teste la fonction load_data
    df = load_data(sample_data)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_data(sample_data):
    # Teste la fonction preprocess_data
    X, y = preprocess_data(sample_data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == 3  # 3 colonnes après get_dummies
    assert 'Sex_male' in X.columns  # La colonne créée par get_dummies

def test_split_data(sample_data):
    # Teste la fonction split_data
    X, y = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

def test_scale_data(sample_data):
    # Teste la fonction scale_data
    X, y = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

def test_train_model(sample_data):
    # Teste la fonction train_model
    X, y = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    assert isinstance(model, LogisticRegression)

def test_evaluate_model(sample_data):
    # Teste la fonction evaluate_model
    X, y = preprocess_data(sample_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    results = evaluate_model(model, X_test_scaled, y_test)
    assert isinstance(results, tuple)
    assert len(results) == 5  # Accuracy, Precision, Recall, F1, Confusion Matrix

if __name__ == '__main__':
    pytest.main()