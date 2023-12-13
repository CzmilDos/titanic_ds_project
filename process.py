import pandas as pd
import seaborn as sns


def main():
    """
    Main
    """
    data = import_data()
    data_cleaning(data)



def import_data():
    """
    Import csv file as a dataframe
    Output: data [pd.DataFrame]
    """
    titanic =  sns.load_dataset('titanic')
    print(titanic.shape)
    return titanic


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perfom cleaning operations
    on received dataframe
    """
    # Vérification de nombres de valeurs manquantes par colonne
    data.isnull().sum()

    # Suppression de la colonne deck (688 valeurs manquantes sur 891)
    data = data.drop(columns=['deck'], axis=1)

    # Remplacement des valeurs manquantes de age par l'age moyenne
    data['age'] = data['age'].fillna(data['age'].mean())

    # remplacements valeurs manquantes de embarked par le mode
    data['embarked'].mode()
    data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

    # remplacer manquantes de embark_town par le mode
    data['embark_town'].mode()
    data['embark_town'] = data['embark_town'].fillna(data['embark_town'].mode()[0])

    # Arrondissement prix billet et conversion age
    data['fare'] = data['fare'].round(2)
    data['age'] = data['age'].astype(int)

    return data


if __name__ == '__main__':
    """
    Doc
    """
    main()
