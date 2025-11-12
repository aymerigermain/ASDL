import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm

# On charge les données
data = np.loadtxt("defautsrails.dat")
X = data[:,:-1] # tout sauf la dernière colonne
y = data[:,-1] # uniquement la dernière colonne


# Exercice 3

# 3.1
# Dans un nouveau programme rails.py, écrivez une boucle pour créez les 4 classifieurs binaires
# SVM linéaires nécessaires à la méthode de décomposition un-contre-tous.
# 2. Complétez la boucle pour calculer, pour chaque classifieur binaire séparément, son taux de recon-
# naissance sur la base d’apprentissage.

# On prend Y et on requalifie de manière binaire
# Ylist[i] contiendra les labels binaires pour la classe i
models = [None] * 4 # liste de 4 modèles
ylist = [[] for _ in range(4)]
for i in range(4):
    Y = 2*(y==i+1)-1 # 1 si classe i, -1 sinon
    ylist[i].append(Y)
    # Classification "un contre tous" pour 4 classes
    models[i] = svm.LinearSVC(C=10, max_iter=10000) # max_iter pour éviter les warnings
    models[i].fit(X, np.ravel(ylist[i])) # np.ravel pour transformer en vecteur 1D depuis une matrice Nx1
    # Taux de reconnaissance
    ypred = models[i].predict(X)
    taux = np.sum(ypred == ylist[i][0]) / len(y)
    print("Taux de reconnaissance pour la classe ", i+1, " : ", taux)


# 3.2 Combinaison des classifieurs binaires
# Complétez rails.py pour calculer la prédiction multi-classe à partir des classifieurs binaires et
# calculez l’erreur d’apprentissage du classifieur multi-classe global

# Aide :
# indiceDuMax = np.argmax(u) # pour un vecteur u
# indicesColonneMax = np.argmax( U, axis=0 ) # pour une matrice U

# On crée un matrice G de scores
G = np.zeros((len(y), 4))
for i in range(4):
    G[:, i] = models[i].decision_function(X)


# On boucle sur les exemples pour faire la prédiction multi-classe
for i in range(len(y)):
    print("Exemple ", i, " : scores = ", G[i, :], " => classe prédite = ", np.argmax(G[i, :]) + 1)

# Prédiction multi-classe
ypred_multi = np.argmax(G, axis=1) + 1 # +1 car les classes sont 1,2,3,4 et pas 0,1,2,3
# axis=1 pour chercher le max sur les lignes
# Pour chaque ligne, on regarde la colonne qui a la plus grande valeur
# En effet, la classe prédite par ce classifieur correspond au classifieur binaire conduisant à
# la valeur réelle maximale en sortie
# la fonction argmax renvoie l'indice de la valeur maximale
# ypred_multi est un vecteur de taille len(y) contenant les classes prédites


# Calcul de l'erreur d'apprentissage
taux_multi = np.mean(ypred_multi == y)
print("Taux de reconnaissance multi-classe : ", taux_multi)

# 3.3 Estimation de l’erreur de généralisation par validation croisée
# Le faible nombre d’exemples disponibles ne permet pas d’en retenir pour créer une base de test indé-
# pendante de la base d’apprentissage. Pour évaluer les performances en généralisation du classifieur, nous
# allons donc utiliser la technique de validation croisée Leave-One-Out (LOO).
# Complétez le main pour réaliser les tâches suivantes.

# L’apprentissage du classifieur multi-classe (et donc des 4 classifieurs binaires) de l’exercice pré-
# cédent doit être répété pour chaque i de 0 à 139 avec l’exemple d’indice i exclu de la base d’ap-
# prentissage. Pour visualiser l’évolution de la procédure (qui peut être un peu longue), vous pouvez
# éventuellement afficher i à chaque itération. Il n’est pas nécessaire de mémoriser tous les classifieurs
# obtenus (voir question suivante).
# On peut utiliser X_i = np.delete(X, i, axis=0) pour récupérer une copie de X sans la
# ième ligne et y_i = np.delete(y, i) pour un vecteur.

n = len(y) # nombre d'exemples, ici 140
ypred_loo = np.zeros(n)

# On boucle sur les exemples
for i in range(n):
    print("Itération ", i)
    X_i = np.delete(X, i, axis=0)
    y_i = np.delete(y, i)

    
    # Répéter l'apprentissage du classifieur multi-classe ici
    models = [None] * 4 # liste de 4 modèles
    ylist = [[] for _ in range(4)]

    # On boucle sur les classes pour créer les classifieurs binaires (comme en 3.1)
    for j in range(4):
        Y = 2*(y_i==j+1)-1 # 1 si classe j, -1 sinon
        ylist[j].append(Y)
        # Classification "un contre tous" pour 4 classes
        models[j] = svm.LinearSVC(C=10, max_iter=10000) # max_iter pour éviter les warnings
        models[j].fit(X_i, np.ravel(ylist[j])) # np.ravel pour transformer en vecteur 1D depuis une matrice Nx1
        # Taux de reconnaissance
        ypred = models[j].predict(X_i)
        taux = np.sum(ypred == ylist[j][0]) / len(y_i)
        print("Taux de reconnaissance pour la classe ", j+1, " : ", taux)

    # Prédiction multi-classe
    G = np.zeros((len(y_i), 4))
    for j in range(4):
        G[:, j] = models[j].decision_function(X_i)

    ypred_multi = np.argmax(G, axis=1) + 1 # +1 car les classes sont 1,2,3,4 et pas 0,1,2,3
    # Calcul de l'erreur d'apprentissage
    taux_multi = np.mean(ypred_multi == y_i)
    print("Taux de reconnaissance multi-classe : ", taux_multi)

    # Stocker la prédiction pour l'exemple i
    ypred_loo[i] = ypred_multi[0]