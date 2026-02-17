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

"""
def verification_bornes(valeurs_in):
    x1,x2,x3 = valeurs_in
    if (0.05< x1 < 2.0) and (0.25 < x2 < 1.3) and (2.0 < x3 < 15.0) :
        return True
    else :
        return False
"""
def penaliser_algo():
    pass

def main():
    n=50

    algo = RechercheAleatoire(n, fonction_objectives)
    meilleur_x, meilleur_solution, history = algo.run()
    print("Meilleure solution : ", meilleur_x)
    print("Meilleure valeur : ", meilleur_solution)
    print("Historique : ", history)



if __name__ == "__main__":    
    main()
    