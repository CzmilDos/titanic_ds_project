"""
Tests
"""
import pandas as pd
from process import import_data 
from process import data_cleaning


def test_import_data():
    """
    Teste la fonction import_data
    """
    data = import_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


def test_data_cleaning():
    """
    Teste la fonction data_cleaning
    """
    data = import_data()
    cleaned_data = data_cleaning(data)

    # Vérification du nombre de valeurs manquantes après nettoyage
    assert cleaned_data.isnull().sum().sum() == 0

    # Vérification de la suppression de la colonne 'deck'
    assert 'deck' not in cleaned_data.columns

    # Vérification de la conversion de 'age' en entiers
    assert cleaned_data['age'].dtype == int

    # Vérification de l'arrondissement du prix du billet à deux chiffres après la virgule
    assert all(abs(cleaned_data['fare'] % 0.01) < 1e-10)

