                #################################################################################
                #                              BIBLIOTHEQUE                                     #
                #################################################################################
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import seaborn as sns

df = pd.read_csv("C:/Users/makan/OneDrive/Bureau/projet Data/bank.csv")
pd.set_option('display.max_columns', None)
df.head()
df.describe()  # bcp de valeurs extrême dans les valeurs numériques
df.info()

                #################################################################################
                #                    1. ANALYSE EXPLORATOIRES DES DONNEES                       #
                #################################################################################

col_num = df.select_dtypes(include='int64')
for col in col_num:
    fig = px.histogram(df, df[col],
                 title = f'Distribution de {col}', histnorm = 'density')
    fig.show()
    print(f"La kutosis de {col} est égale à : {stats.kurtosis(df[col])}") #pour avoir une idée de son éloignement par rapport a la loi normale

col_cat = df.select_dtypes(include='object')
for col in col_cat:
    fig = px.histogram(df, x = df[col], histnorm='percent',
           title = f'Distribution de {col}')
    fig.show()

                # ###############################################################################
                #                         2. TRANSFORMATION DES DONNÉES                         #
                # ###############################################################################

#On a pu visualiser la présence de valeurs extrêmes au travers de la section précédentes

# Application des transformation aux var numériques
transformation = ['age', 'yj_balance', 'log_duration']
df['log_duration'] = np.log(df['duration']) + 1 # car la valeurs tjrs >= 0

pt = PowerTransformer(method='yeo-johnson') #utilisation de la transformation yj car valeurs négative
df['yj_balance'] = pt.fit_transform(pd.DataFrame(winsorize(df['balance'], limits=[0.01, 0.01])))
df['yj_pdays'] = pt.fit_transform(pd.DataFrame(df['pdays']))

df.drop(['balance', 'duration'], axis=1, inplace=True)

bin = [-1, 0, 49, 120, 160, 209, 249, 319, 399, 854]
lab = ["-1", "0-49", "50-120", "121-160", "161-209", "210-249", "250-319", "320-399", "400+"]
df['int_pdays'] = pd.cut(df['pdays'], bins=bin, labels=lab, include_lowest=True)

for col in transformation:
    fig = px.histogram(df, x = df[col],
                        title = f"Distribution de {col} après transformation")
    fig.show()
    print(f"La kurtosis de {col} est : ", stats.kurtosis(df[col]), "et la skewness est :", stats.skew(df[col]))

# QQPLOT : comparaison a une distribution normale
for col in transformation:
    plt.figure()
    stats.probplot(df[col], dist='norm', plot=plt)
    plt.title(f"QQ-plot de {col}")
    plt.show()


# Création de variables
def tranches_ages(age):
    if age <= 30:
        return 'Jeune Adulte'
    elif age <= 45:
        return 'Adulte'
    elif age <= 65:
        return 'Senior'
    else:
        return 'Senior +'

df['Tranche'] = df['age'].apply(tranches_ages)

                # ###############################################################################
                #                        3. SÉLECTION DES VARIABLES                             #
                # ###############################################################################

# A. Sélection univariée
df['deposit'] = df['deposit'].apply(lambda x: 1 if x == 'yes' else 0)
y = df['deposit'].astype('category')

X = df.drop("deposit", axis=1)
for col in X:
    if X[col].dtype == "object":
        X[col] = X[col].astype('category')

from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
X_encod = pd.get_dummies(X, drop_first=True)
selector = SelectKBest(mutual_info_classif, k=10) # utilisation de mutual_info_classif car mixte var cate et cont pas forcément normales
X_ten_select = selector.fit_transform(X_encod, y)

slectionnees = X_encod.columns[selector.get_support()]
print("Les 10 variables sélectionnées sont : ", slectionnees.tolist())
print("Scores des variables sélectionnées : ", selector.scores_[selector.get_support()])
tableau = pd.DataFrame({'variables': slectionnees.tolist(), 'Score': selector.scores_[selector.get_support()]}).sort_values(by='Score', ascending=False).reset_index(drop=True)
print(tableau)
# print(df.head())

# B.Sélection multivariée

X_continue = X_encod[slectionnees.tolist()].select_dtypes(include=['int64', 'float'])
for col in X_continue:
    corr, pvalue = stats.pearsonr(X_continue[col], y)
    print(f'Corrélation entre {col} et la variable cible est : {corr}.\n la p_value est égale à {pvalue}.')

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X_ten_select_df = pd.DataFrame(X_ten_select, columns=slectionnees.tolist())
X_ten_select_df = add_constant(X_ten_select_df)

#Calcul du VIF : vérification pas de colinéarité dans les var selectionnées
VIF = pd.DataFrame()
VIF['variable'] = X_ten_select_df.columns
VIF['calcul_Vif'] = [variance_inflation_factor(X_ten_select_df.values, i) for i in range(X_ten_select_df.shape[1])]
print(VIF)

sns.heatmap(X_ten_select_df.drop("const", axis=1).corr(), annot=True, fmt=".2f")
plt.show()
#poutcome_unknow et pdays/yj-pdays forte colinéarité

# Test ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('yj_pdays ~ poutcome', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n ", anova_table, "\n")

#Les variables qu'on retient : ["yj-pdays","log_duration", "previous"]


# Cramer'V : donne la puissance de la relation MAIS sans indiquer le sens
matrice = []

def Cramer():
    global matrice
    n = len(col_cat)
    liste = []
    for col1 in col_cat:
        for col2 in col_cat:
            table = pd.crosstab(df[col1], df[col2])
            chi_stat = stats.chi2_contingency(table)[0]
            k = min(len(df[col1].unique()), len(df[col2].unique())) - 1
            v_cram = np.sqrt((chi_stat)/(n*k))
            liste.append(v_cram)
        matrice.append(liste)
        liste = []
    matrice = pd.DataFrame(matrice, columns=col_cat.columns.tolist(), index=col_cat.columns.tolist())
    return matrice

Cramer()
sns.heatmap(matrice, vmin=0, vmax=1, annot=True, fmt='.2f')
plt.show()

#X_ten_select_df.drop(['poutcome_unknown','pdays', 'previous'])
print(X_ten_select_df.columns.tolist())
X_ten_select_df.drop(['pdays', 'previous'], axis=1)

                    # ###############################################################################
                    #                          4. PREPARATION AU MODELE                             #
                    # ###############################################################################

from sklearn.feature_selection import SequentialFeatureSelector # forward selection
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb_model = XGBClassifier()
selection = SequentialFeatureSelector(xgb_model, scoring='f1', n_features_to_select=4, direction='forward', cv=5)
selection.fit(X_ten_select_df, y)
var_selection = X_ten_select_df.columns[selection.get_support()]
print(f"Les variables sélectionnées sont : ", var_selection.tolist())

X_ten_select_df = X_ten_select_df[var_selection]

#Division train/test/eval
np.random.seed(2003)
ind_eval = np.random.choice(np.arange(len(X_ten_select_df)), size=int(0.2 * len(X_ten_select_df)), replace=False)
X_eval = X_ten_select_df.iloc[ind_eval]
y_eval = y.iloc[ind_eval]

qvpp = np.setdiff1d(np.arange(len(X_ten_select_df)), ind_eval)
X_restant = X_ten_select_df.iloc[qvpp]
y_restant = y.iloc[qvpp]

X_train, X_test, y_train, y_test = train_test_split(X_restant, y_restant, test_size=0.2, random_state=42, stratify=y_restant)

# Hyper-paramètres après gried search sur pls hyper paramètees
params = {
    'objective': ['binary:logistic'],
    'max_depth': [5],
    'learning_rate': [0.01],
    'n_estimators': [300]
}

                # ###############################################################################
                #                      5. MODELE DE PREDICTION : XGBOOST                        #
                # ###############################################################################


grid_model = GridSearchCV(
    estimator=xgb_model,
    param_grid=params,
    cv=5,
    scoring='f1_macro'
)

grid_model.fit(X_train, y_train)
print("Meilleurs paramètres :", grid_model.best_params_)

xgb_model1 = XGBClassifier(**grid_model.best_params_, random_state=42)
xgb_model1.fit(X_train, y_train)

y_pred_train = xgb_model1.predict(X_train)
y_pred_test = xgb_model1.predict(X_test)
y_pred_eval = xgb_model1.predict(X_eval)

#Matrices de confusion
confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
confusion_matrix_eval = confusion_matrix(y_eval, y_pred_eval)

confusion_matrix_train_percent = confusion_matrix_train.astype('float')/confusion_matrix_train.sum(axis=1)[:, np.newaxis] * 100
confusion_matrix_test_percent = confusion_matrix_test.astype('float')/confusion_matrix_test.sum(axis=1)[:, np.newaxis] * 100
confusion_matrix_eval_percent = confusion_matrix_eval.astype('float')/confusion_matrix_eval.sum(axis=1)[:, np.newaxis] * 100

labels = [0, 1]

plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix_train_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matrice de confusion - Train (%)')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

plt.subplot(1, 3, 2)
sns.heatmap(confusion_matrix_test_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matrice de confusion - Test (%)')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')

plt.subplot(1, 3, 3)
sns.heatmap(confusion_matrix_test_percent, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matrice de confusion - Eval (%)')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.tight_layout()
plt.show()

labels_str = ['0', '1']
print("/nF1-Scores (macro):")
print(f"F1-Score (Train): {f1_score(y_train, y_pred_train, average='macro'):.3f}")
print(f"F1-Score (Test): {f1_score(y_test, y_pred_test, average='macro'):.3f}")
print(f"F1-Score (Évaluation): {f1_score(y_eval, y_pred_eval, average='macro'):.3f}")

print("/nRapport de classification - Train:")
print(classification_report(y_train, y_pred_train, target_names=labels_str))
print("/nRapport de classification - Test:")
print(classification_report(y_test, y_pred_test, target_names=labels_str))
print("/nRapport de classification - Évaluation:")
print(classification_report(y_eval, y_pred_eval, target_names=labels_str))
