import numpy as np
import matplotlib.pyplot as plt
from recherche_aleatoire import RechercheAleatoire

def fonction_objectives(valeurs_in):
    x1,x2,x3 = valeurs_in
    return (x1 ** 2)* x2 * (2+x3)

def contraintes_fonction(valeurs_in):
    x1,x2,x3 = valeurs_in
    g1 = 1 - (x2 ** 3 ) * x3 / 71785 * x1 ** 4 <= 0
    g2 = 4 * (x2 ** 2 ) - x1 * x2 / 12566 * (x2 * (x1 **3) - x1** 4) <= 0
    g3 = 1 -(140.5 * x1 / ((x2 **2) * x3)) <= 0
    g4 = ((x1 + x2 ) /1.5)-1 <= 0
    return g1,g2,g3,g4

def verification_contraintes(valeurs_in):
    g1,g2,g3,g4 = contraintes_fonction(valeurs_in)
    if g1 and g2 and g3 and g4 :
        return True
    else :
        return False

"""
def verification_bornes(valeurs_in):
    x1,x2,x3 = valeurs_in
    if (0.05< x1 < 2.0) and (0.25 < x2 < 1.3) and (2.0 < x3 < 15.0) :
        return True
    else :
        return False
"""
def penaliser_algo(valeurs_in):
    penalite = 10
    fonction = fonction_objectives(valeurs_in)

    if verification_contraintes(valeurs_in) == False :
        penalite += 10
        return fonction + penalite
    else :
        return fonction

        

def main():
    n=100

    algo = RechercheAleatoire(n, penaliser_algo)
    meilleur_x, meilleur_solution, history = algo.run()
    print("Meilleure solution : ", meilleur_x)
    print("Meilleure valeur : ", meilleur_solution)
    print("Historique : ", history)

    plt.plot( [h[1] for h in history])
    plt.title('Recherche Aléatoire')
    plt.show()



if __name__ == "__main__":    
    main()
    