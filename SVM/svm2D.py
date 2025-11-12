import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm



#### programme principal à implémenter dans cette fonction ####
def monprogramme(Xapp, Yapp, C):
	"""
		Programme pour les SVM linéaires (lancé avec ESPACE)
		
		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
	"""
	print("Apprentissage lancé avec " + str(len(Xapp)) + " points et C = ", C)

	# à compléter pour apprendre le modèle SVM...

	# # SVM à marge dure 
	# C = math.inf
	# # On oblige à avoir une erreur à 0

	# # SVM à marge souple
	# C = 10
	model = svm.LinearSVC(C=C)
	model.fit(Xapp, Yapp)


	# création d'une grille de points de test
	r1 = np.arange(-5,5,0.2)
	# création de la matrice de test
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1
	
	# Prédire la catégorie pour tous les points de test...

	Ypred = model.predict(Xtest)

	# ... et tracer le résultat avec par exemple 

	Xbleu = Xtest[Ypred==1,:]
	Xrouge = Xtest[Ypred==2,:]
	plt.plot(Xbleu[:,0], Xbleu[:,1], 'ob', alpha=0.2) # points bleus (abscisses, ordonnées), ob pour cercles bleus, alpha pour transparence
	plt.plot(Xrouge[:,0], Xrouge[:,1], 'or', alpha=0.2) # points rouges
	# for i in range(len(Xtest)):
	# 	plt.plot(Xtest[i, 0], Xtest[i, 1], '.b' if Ypred[i] == 1 else '.r')
	
	
	# tracer la droite séparation et les marges... 
	w = model.coef_[0]
	b = model.intercept_

	# Il nous faut 2 points
	# Un point p de coordonnées (p1, p2) 
	# Un point q de coordonnées (q1, q2)
	# wTp + b = 0 les points sur la droite
	# <=> w1p1 + w2p2 + b = 0 
	# <=> p2 = -(w1/w2)p1 - b/w2
	# On prend p1 = 5 et q1 = 5
	w1 = w[0]
	w2 = w[1]
	p1 = -5
	q1 = 5
	p2 = -(w1/w2)*p1 - b/w2
	q2 = -(w1/w2)*q1 - b/w2
	plt.plot([p1, q1], [p2, q2], 'k-') # droite de séparation

	
	

	# calculer et afficher la marge Delta...	

	# Les point sur la marge satisfont |wTx + b| = 1
	# <=> wTx + b = 1 ou wTx + b = - 1
	# <=> w1x1 + w2x2 + b = 1 ou w1x1 + w2x2 + b = -1
	# <=> x2 = -(w1/w2)x1 - (b-1)/w2 ou x2 = -(w1/w2)x1 - (b+1)/w2
	# On prend x1 = 5 et y1 = 5
	p1 = -5
	q1 = 5
	p2 = -(w1/w2)*p1 - (b-1)/w2
	q2 = -(w1/w2)*q1 - (b-1)/w2
	plt.plot([p1, q1], [p2, q2], 'k--') # marge 1
	p2 = -(w1/w2)*p1 - (b+1)/w2
	q2 = -(w1/w2)*q1 - (b+1)/w2
	plt.plot([p1, q1], [p2, q2], 'k--') # marge 2


	# EN changeant la valeur de C, la marge s'élargit ou se réduit
	# Avec un grand C, marge petite
	# Avec un petit C, marge grande

	
	# pour réellement mettre à jour le graphique: (à garder en fin de fonction)
	fig.canvas.draw()


def monprogrammeNL(Xapp, Yapp, C, sigma):
	"""
		Programme pour les SVM non linéaires (lancé avec N)
		
		Xapp, Yapp : base d'apprentissage générée avec la souris
		C	: paramètre réglé par +/-
		sigma : paramètre réglé par CTRL +/-
	"""
	print("Apprentissage lancé avec " + str(len(Xapp)) + " points, C = ", C, " et sigma = ", sigma )

	# à compléter pour apprendre le modèle SVM non linéaire...
	model = svm.SVC(C=C, kernel='rbf', gamma=1/(2*sigma**2))
	model.fit(Xapp, Yapp)

	# création d'une grille de points de test
	r1 = np.arange(-5,5,0.2)
	Xtest = np.zeros((len(r1)*len(r1),2))
	i = 0
	for x1 in r1:
		for x2 in r1:
			Xtest[i,:] = [x1, x2]
			i += 1
	
	# Prédire la catégorie pour tous les points de test...
	Ypred = model.predict(Xtest)

	
	# ... et tracer le résultat avec par exemple 
	# Xbleu = Xtest[Ypred==1,:]
	# Xrouge = Xtest[Ypred==2,:]
	# plt.plot(Xbleu[:,0], Xbleu[:,1], 'ob', alpha=0.2) # points bleus (abscisses, ordonnées)
	# plt.plot(Xrouge[:,0], Xrouge[:,1], 'or', alpha=0.2) # points rouges

	# f(x) = sign(g(x)) avec g(x) = model.decision_function(x)
	# Frontière de décision : g(x) = 0
	# Marges : g(x) = 1 et g(x) = -1 cad |g(x)| = 1
	# On colorie en Rouge / Rose / Bleu clair / Bleu foncé selon |g(x)| < 1 ou > 1 et g(x) > 0 ou < 0
	gtest = model.decision_function(Xtest)
	Xrougefonce = Xtest[(gtest < -1), :]
	Xrose = Xtest[(gtest >= -1) & (gtest < 0), :]
	Xbleuclair = Xtest[(gtest >= 0) & (gtest < 1), :]
	Xbleufonce = Xtest[(gtest >= 1), :]
	plt.plot(Xrougefonce[:,0], Xrougefonce[:,1], 'ob', alpha=0.2) # points rouges foncés
	plt.plot(Xrose[:,0], Xrose[:,1], 'o', color=(0.5,0.5,1), alpha=0.2) # points roses
	plt.plot(Xbleuclair[:,0], Xbleuclair[:,1], 'o', color=(1,0.5,0.5), alpha=0.2) # points bleu clair
	plt.plot(Xbleufonce[:,0], Xbleufonce[:,1], 'or', alpha=0.2) # points bleu foncés
	



	# afficher le nombre de vecteurs support...	
	print(model.n_support_)

	# Plus sigma est petit, plus il y a de vecteurs support car la frontière est plus complexe
	# Plus C est grand, plus il y a de vecteurs support car la marge est plus petite

	
	
	# pour réellement mettre à jour le graphique: 
	fig.canvas.draw()
	



##### Gestion de l'interface graphique ########


Xplot = np.zeros((0,2))
Yplot = np.zeros(0)
plotvariance = 0

C = 1
sigma = 1

def onclick(event):
	global Xplot
	global Yplot
	
	
	if plotvariance == 0:
		newX = np.array([[event.xdata,event.ydata]])
	else:
		newX = math.sqrt(plotvariance) * np.random.randn(10, 2) + np.ones((10,1)).dot(np.array([[event.xdata,event.ydata]]))

	print("Ajout de " + str(len(newX)) + " points en (" + str(event.xdata) + ", " + str(event.ydata) + ")")

	Xplot = np.concatenate((Xplot,newX))
	if event.button == 1 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.b')
		newY = np.ones(len(newX)) * 1
	elif event.button == 3 and event.key == None:
		plt.plot(newX[:,0], newX[:,1],'.r')
		newY = np.ones(len(newX)) * 2
	Yplot = np.concatenate((Yplot,newY))
	
	fig.canvas.draw()


def onscroll(event):
	global plotvariance
	if event.button == "up":
		plotvariance = round(plotvariance + 0.2, 1)
	elif event.button == "down" and plotvariance > 0.1:
		plotvariance = round(plotvariance - 0.2, 1)
	print("Variance = ", plotvariance)

def onkeypress(event):
	global C
	global sigma
	if event.key == " ":
		monprogramme(Xplot, Yplot, C)
	elif event.key == "n":
		monprogrammeNL(Xplot, Yplot, C, sigma)
	elif event.key == "+":
		C *= 2
		print("C = " , C)
	elif event.key == "-":
		C /= 2
		print("C = " , C)
	elif event.key == "ctrl++":
		sigma *= 2
		print("sigma = " , sigma)
	elif event.key == "ctrl+-":
		sigma /= 2
		print("sigma = " , sigma)
				
	
fig = plt.figure()

plt.axis([-5, 5, -5, 5])

cid = fig.canvas.mpl_connect("button_press_event", onclick)
cid2 = fig.canvas.mpl_connect("scroll_event", onscroll)
cid3 = fig.canvas.mpl_connect("key_press_event", onkeypress)

print("Utilisez la souris pour ajouter des points à la base d'apprentissage :")
print(" clic gauche : points bleus")
print(" clic droit : points rouges")
print("\nMolette : +/- variance ")
print("   si variance = 0  => ajout d'un point")
print("   si variance > 0  => ajout de points selon une loi gaussienne")
print("\n ESPACE pour lancer la fonction monprogramme(Xapp,Yapp,C)")
print("    avec la valeur de C modifiée par +/-") 
print("\n N pour lancer la fonction monprogrammeNL(Xapp,Yapp,C,sigma)")
print("    avec la valeur de C modifiée par +/-")
print("    et celle de sigma modifiée par CTRL +/-\n\n") 

plt.show()
