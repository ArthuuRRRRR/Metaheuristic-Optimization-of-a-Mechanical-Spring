import numpy as np
import matplotlib.pyplot as plt
from random_search import RandomSearch
from Hill_Climbing_simple_1_1 import Hill_Climbing_1_1  
from generalized_hill_climbing import generalized_hill_climbing
from creer_le_pickle import creer_pickle
import pandas as pd
from simulated_annealing import SimulatedAnnealing
from display_result import display_result

def fonction_objectives(valeurs_in):
    x1,x2,x3 = valeurs_in
    return (x1 ** 2)* x2 * (2+x3)

def contraintes_fonction(valeurs_in):
    x1,x2,x3 = valeurs_in
    g1 = 1 - ((x2 ** 3 ) * x3) / (71785 * x1 ** 4) 
    g2 = (4 * (x2 ** 2 ) - x1 * x2) / (12566 * (x2 * (x1 **3) - x1** 4)) + (1/ (5108 * (x1**2))) - 1 
    g3 = 1 -(140.45 * x1 / ((x2 **2) * x3)) 
    g4 = ((x1 + x2 ) /1.5)-1 
    return g1,g2,g3,g4


def verification_contraintes(valeurs_in):
    g1,g2,g3,g4 = contraintes_fonction(valeurs_in)
    if g1 > 0 and g2 > 0 and g3 > 0 and g4 > 0:
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
        penalite += penalite *1.5
        return fonction + penalite
    else :
        return fonction
    

def penaliser_algo_2(valeurs_in):
    f = fonction_objectives(valeurs_in)

    g1, g2, g3, g4 = contraintes_fonction(valeurs_in)

    v1 = max(0.0, g1)
    v2 = max(0.0, g2)
    v3 = max(0.0, g3)
    v4 = max(0.0, g4)

    violation_totale = v1 + v2 + v3 + v4

    coef_penalite = 1000  
    penalite = coef_penalite * violation_totale

    return f + penalite


def main():
    n=1000
    variable = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]

    algo_recherche_aleatoire = RandomSearch(n, penaliser_algo_2)
    meilleur_x, meilleur_solution, history = algo_recherche_aleatoire.run()
    print("Meilleure solution : ", meilleur_x)
    print("Meilleure valeur : ", meilleur_solution)


    algo_hill_climbing1_1 = Hill_Climbing_1_1(n, penaliser_algo_2, variable, 0.05)
    meilleur_x_hc, meilleur_solution_hc, history_hc = algo_hill_climbing1_1.run()

    
    print("Meilleure solution Hill Climbing : ", meilleur_x_hc)
    print("Meilleure valeur Hill Climbing : ", meilleur_solution_hc)

    algo_hill_climbing = generalized_hill_climbing(n, penaliser_algo_2, variable, 0.05, 8)
    meilleur_x_hc_G, meilleur_solution_hc_G, history_hc_G = algo_hill_climbing.run()

    print("Meilleure solution Hill Climbing : ", meilleur_x_hc_G)
    print("Meilleure valeur Hill Climbing : ", meilleur_solution_hc_G)

    algo_annealing = SimulatedAnnealing(n, penaliser_algo_2, variable, 0.05,"exponentielle")
    meilleur_x_annealing, meilleur_solution_annealing, history_annealing = algo_annealing.run()
    print("Meilleure solution Simulated Annealing : ", meilleur_x_annealing)
    print("Meilleure valeur Simulated Annealing : ", meilleur_solution_annealing)

    """
    df = creer_pickle(
    penaliser_algo=penaliser_algo_2,
    variable=variable,
    n_iter=n,
    nb_simulations=100,
    pas=0.05,
    nbr_voisin=8,
    outfile="resultats.pkl",
    outcsv="resultats.csv",
    )
    print("Pickle créé:", "resultats.pkl")"""


    PATH = r"C:\Users\delha\OneDrive\Desktop\Cours_UQAR\Metaheuristique\resultats.csv"

    df = display_result.load_results(PATH)
    agg = display_result.aggregate_convergence(df)

    display_result.plot_all_algos(
        agg,
        algos_order=("RS", "HC", "GHC", "SA"),
        use_log=True,
        show_minmax=False,  
        title="Comparaison des métaheuristiques — médiane + Monte-Carlo",)



if __name__ == "__main__":    
    main()
    