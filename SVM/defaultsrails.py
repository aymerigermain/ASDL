import numpy as np
import matplotlib.pyplot as plt
import math

# On charge les données
data = np.loadtxt("defautsrails.dat")
X = data[:,:-1] # tout sauf la dernière colonne
y = data[:,-1] # uniquement la dernière colonne


from sklearn import svm

# Classification "un contre tous" pour 4 classes
model = [None] * 4 # liste de 4 modèles

# On crée un vecteu
score = np.zeros((len(y), 4))
for k in range(4):
	model[k] = svm.LinearSVC(C=10)	
	model[k].fit(X,2*(y==(k+1))-1)
	
	score[:,k] = model[k].decision_function(X)
print(np.argmax(score,axis=1))
print(model[0].decision_function([ X[0,:] ]), model[0].decision_function([ X[0] ]))
