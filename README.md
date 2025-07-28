# Projet de prédiction de souscription à un terme de dépôt

## Description  
Ce projet de machine learning a été réalisé de manière personnelle à partir d'une base de données trouvée sur Kaggle.  
L'objectif est de prédire, à partir des caractéristiques clients, s'ils vont souscrire à un certain contrat.

## Prérequis  
- Python 3.11  
- Bibliothèques utilisées : pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, xgboost, scipy, category_encoders, imbalanced-learn, joblib  

## Utilisation  
Le code s'exécute simplement en une fois. Plusieurs fenêtres contenant des graphiques s'ouvriront automatiquement pour visualiser les données.

## Structure du projet  
1. Exploration des données avec pandas, plotly et seaborn.  
2. Traitement des valeurs extrêmes via des transformations statistiques (Yeo-Johnson, logarithmique).  
3. Sélection des variables : univariée (SelectKBest) et multivariée (corrélations, V de Cramer) avec scikit-learn.  
4. Préparation et sélection progressive du modèle (forward selection) pour obtenir un modèle simple avec 4 variables.  
5. Entraînement d’un modèle XGBoost avec optimisation des hyperparamètres.

## Résultats  
F1-score autour de 80%, résultat satisfaisant qui pourrait s’améliorer avec plus de données et d’entraînement.

## Auteur  
[@AnneliMks ](https://github.com/AnneliMks)


------------------------------------------------------------------------------------------------------------------
# Deposit Term Subscription Prediction Project

## Description  
This machine learning project was done personally using a dataset found on Kaggle.  
The goal is to predict, based on customer features, whether they will subscribe to a specific contract.

## Requirements  
- Python 3.11  
- Libraries used: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, xgboost, scipy, category_encoders, imbalanced-learn, joblib  

## Usage  
The code runs in one go. Multiple windows with charts will open automatically to visualize the data.

## Project Structure  
1. Data exploration using pandas, plotly, and seaborn.  
2. Handling outliers with statistical transformations (Yeo-Johnson, logarithmic).  
3. Feature selection: univariate (SelectKBest) and multivariate (correlation, Cramér's V) using scikit-learn.  
4. Model preparation with forward selection to build a parsimonious model with 4 variables.  
5. Training an XGBoost model with hyperparameter tuning.

## Results  
F1-score around 80%, a satisfactory result that could be improved with more data and training.

## Author  
[@AnneliMks ](https://github.com/AnneliMks)
