# Prédiction des Maladies Cardiaques

Ce projet vise à prédire la probabilité qu'un patient développe une maladie cardiaque en utilisant un ensemble de données cliniques. Plusieurs modèles d'apprentissage automatique sont explorés et comparés pour déterminer le plus performant.

## Jeu de Données

Le jeu de données utilisé provient de Kaggle : [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

Il contient les caractéristiques suivantes :
*   Age : âge du patient [années]
*   Sex : sexe du patient [M: Homme, F: Femme]
*   ChestPainType : type de douleur thoracique [TA: Angine Typique, ATA: Angine Atypique, NAP: Douleur Non Angineuse, ASY: Asymptomatique]
*   RestingBP : pression artérielle au repos [mm Hg]
*   Cholesterol : cholestérol sérique [mm/dl]
*   FastingBS : glycémie à jeun [1: si FastingBS > 120 mg/dl, 0: sinon]
*   RestingECG : résultats de l'électrocardiogramme au repos [Normal: Normal, ST: ayant une anomalie de l'onde ST-T, LVH: montrant une hypertrophie ventriculaire gauche probable ou définie]
*   MaxHR : fréquence cardiaque maximale atteinte [valeur numérique entre 60 et 202]
*   ExerciseAngina : angine induite par l'exercice [Y: Oui, N: Non]
*   Oldpeak : oldpeak = ST [valeur numérique mesurée en dépression]
*   ST_Slope : la pente du segment ST de l'exercice maximal [Up: ascendante, Flat: plate, Down: descendante]
*   HeartDisease : classe de sortie [1: maladie cardiaque, 0: Normal]

## Étapes du Projet

Le notebook Jupyter ([Predicting_Heart_Disease_1.ipynb](c%3A%5CUsers%5Csamso%5CDownloads%5CPredicting_Heart_Disease_1.ipynb)) suit les étapes suivantes :

1.  **Chargement des Données** : Téléchargement et chargement du jeu de données.
2.  **Analyse Exploratoire des Données (EDA)** :
    *   Statistiques descriptives.
    *   Visualisation de la distribution des variables catégorielles et numériques.
    *   Identification des valeurs manquantes et des anomalies potentielles.
3.  **Nettoyage des Données** :
    *   Traitement des valeurs aberrantes ou incorrectes (par exemple, RestingBP à 0, Cholesterol à 0).
4.  **Sélection des Caractéristiques (Feature Selection)** :
    *   Encodage des variables catégorielles (one-hot encoding).
    *   Analyse de la corrélation entre les caractéristiques.
    *   Utilisation de l'importance des caractéristiques d'un modèle RandomForest pour identifier les prédicteurs les plus pertinents.
    *   Analyse des courbes de dépendance partielle (Partial Dependence Plots - PDP).
5.  **Construction et Évaluation des Modèles** :
    *   **K-Nearest Neighbors (K-NN)** :
        *   Entraînement avec des caractéristiques uniques et multiples.
        *   Mise à l'échelle des caractéristiques (MinMaxScaler).
        *   Réglage des hyperparamètres (GridSearchCV).
    *   **Autres Modèles d'Ensemble et Arbres de Décision** :
        *   Decision Tree
        *   Random Forest
        *   Gradient Boosting
        *   XGBoost
        *   Comparaison des performances (Accuracy, MSE, AUC-ROC, Sensibilité, Spécificité, Précision, F1-Score).
    *   **Réseaux de Neurones** :
        *   MLPClassifier (Perceptron Multi-Couches).
        *   Réseau de neurones séquentiel avec Keras/TensorFlow, incluant la normalisation des données (StandardScaler), des couches Dense, Dropout, BatchNormalization, et l'optimisation des hyperparamètres (régularisation L2, taux de dropout).
6.  **Comparaison Finale des Modèles** :
    *   Visualisation des performances des différents modèles (graphiques à barres pour l'accuracy, courbes ROC, diagramme radar pour les métriques).
    *   Identification du meilleur modèle basé sur les métriques d'évaluation.

## Modèles Implémentés

*   K-Nearest Neighbors (K-NN)
*   Decision Tree Classifier
*   Random Forest Classifier
*   Gradient Boosting Classifier
*   XGBoost Classifier
*   MLPClassifier (Scikit-learn)
*   Réseau de Neurones Séquentiel (Keras/TensorFlow)

## Comment Exécuter

1.  Assurez-vous d'avoir Python et les bibliothèques nécessaires installées. Vous pouvez les installer via pip :
    ```bash
    pip install pandas scikit-learn matplotlib seaborn kagglehub tensorflow imbalanced-learn xgboost tabulate
    ```
2.  Téléchargez le notebook [Predicting_Heart_Disease_1.ipynb](c%3A%5CUsers%5Csamso%5CDownloads%5CPredicting_Heart_Disease_1.ipynb).
3.  Exécutez les cellules du notebook dans un environnement Jupyter (Jupyter Notebook, JupyterLab, VS Code, etc.).
    *   Note : Le notebook utilise `kagglehub` pour télécharger le jeu de données. Vous pourriez avoir besoin de configurer vos identifiants Kaggle si ce n'est pas déjà fait.

## Résultats

Le projet conclut en comparant les performances de tous les modèles entraînés. Le réseau de neurones à 2 couches cachées a montré des performances prometteuses, notamment en termes d'accuracy globale et de F1-Score, après optimisation des hyperparamètres et du seuil de classification. Les détails des performances de chaque modèle sont disponibles dans le notebook.
