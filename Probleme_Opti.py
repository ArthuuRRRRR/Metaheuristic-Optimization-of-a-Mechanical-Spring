import numpy as np
import matplotlib.pyplot as plt
from recherche_aleatoire import RechercheAleatoire
from Hill_Climbing_simple_1_1 import Hill_Climbing_1_1  

def fonction_objectives(valeurs_in):
    x1,x2,x3 = valeurs_in
    return (x1 ** 2)* x2 * (2+x3)

def contraintes_fonction(valeurs_in):
    x1,x2,x3 = valeurs_in
    g1 = 1 - ((x2 ** 3 ) * x3) / (71785 * x1 ** 4) <= 0
    g2 = (4 * (x2 ** 2 ) - x1 * x2) / (12566 * (x2 * (x1 **3) - x1** 4)) + (1/ (5108 * (x1**2))) - 1 <= 0
    g3 = 1 -(140.45 * x1 / ((x2 **2) * x3)) <= 0
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
    penalite = 1000
    fonction = fonction_objectives(valeurs_in)

    if verification_contraintes(valeurs_in) == False :
        penalite += penalite *2
        return fonction + penalite
    else :
        return fonction

        

def main():
    n=100
    variable = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]

    algo_recherche_aleatoire = RechercheAleatoire(n, penaliser_algo)
    meilleur_x, meilleur_solution, history = algo_recherche_aleatoire.run()
    print("Meilleure solution : ", meilleur_x)
    print("Meilleure valeur : ", meilleur_solution)
    #print("Historique : ", history)

    algo_hill_climbing = Hill_Climbing_1_1(n, penaliser_algo, variable, 0.01)
    meilleur_x_hc, meilleur_solution_hc, history_hc = algo_hill_climbing.run()

    print("Meilleure solution Hill Climbing : ", meilleur_x_hc)
    print("Meilleure valeur Hill Climbing : ", meilleur_solution_hc)

    plt.plot( [h[1] for h in history], label='Recherche Aléatoire')
    plt.plot( [h[1] for h in history_hc], label='Hill Climbing')
    plt.title('Comparaison des algorithmes')
    plt.legend()
    plt.show()



if __name__ == "__main__":    
    main()
    