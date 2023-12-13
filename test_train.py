import unittest
import pandas as pd

from your_module_name import Train

from sklearn.model_selection import train_test_split
from your_module_name import load_data, preprocess_data, split_data, scale_data, train_model, evaluate_model
from train import 

class TestTrain(unittest.TestCase):

    def setUp(self):
        # Charger des données de test
        self.test_data = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 2],
            'Sex': ['male', 'female', 'male', 'female', 'male'],
            'Age': [30, 25, 35, 28, 40],
            'Survived': [1, 0, 1, 0, 1]
        })

    def test_load_data(self):
        # Testez si la fonction load_data charge correctement les données
        df = Train().load_data(self.test_data)
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_preprocess_data(self):
        # Testez si la fonction preprocess_data renvoie les features (X) et les étiquettes (y) correctement
        X, y = Train().preprocess_data(self.test_data)
        self.assertTrue(isinstance(X, pd.DataFrame))
        self.assertTrue(isinstance(y, pd.Series))

    def test_split_data(self):
        # Testez si la fonction split_data divise correctement les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = Train().split_data(self.test_data)
        self.assertTrue(isinstance(X_train, pd.DataFrame))
        self.assertTrue(isinstance(X_test, pd.DataFrame))
        self.assertTrue(isinstance(y_train, pd.Series))
        self.assertTrue(isinstance(y_test, pd.Series))

    def test_scale_data(self):
        # Testez si la fonction scale_data normalise correctement les données
        X_train, X_test = Train().scale_data(self.test_data, self.test_data)
        self.assertTrue(isinstance(X_train, pd.DataFrame))
        self.assertTrue(isinstance(X_test, pd.DataFrame))

    def test_train_model(self):
        # Testez si la fonction train_model entraîne correctement le modèle LogisticRegression
        model = Train().train_model(LogisticRegression(), self.test_data, self.test_data['Survived'])
        # Ajoutez des assertions pour vérifier que le modèle a été correctement entraîné

    def test_evaluate_model(self):
        # Testez si la fonction evaluate_model évalue correctement le modèle LogisticRegression
        model = Train().train_model(LogisticRegression(), self.test_data, self.test_data['Survived'])
        results = Train().evaluate_model(model, self.test_data, self.test_data['Survived'])
        # Ajoutez des assertions pour vérifier les résultats

if __name__ == '__main__':
    unittest.main()
