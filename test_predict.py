import pandas as pd
from sklearn.externals import joblib
import pytest
from predict import load_data, preprocess_data


@pytest.fixture
def sample_model():
    # Charger un modèle fictif pour les tests
    model = joblib.load('trained_model.pkl')
    return model


@pytest.fixture
def sample_scaler():
    # Charger un scaler fictif pour les tests
    scaler = joblib.load('scaler.pkl')
    return scaler


def test_load_data():
    # Teste la fonction load_data
    data = load_data('data/test.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_preprocess_data():
    # Teste la fonction preprocess_data
    data = pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'male'],
        'Age': [25, 30, 35]
    })
    X = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert X.shape[1] == 3


if __name__ == '__main__':
    pytest.main()
