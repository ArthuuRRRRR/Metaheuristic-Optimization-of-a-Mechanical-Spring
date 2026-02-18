import numpy as np
import matplotlib.pyplot as plt
from random_search import RandomSearch
from Hill_Climbing_simple_1_1 import Hill_Climbing_1_1  
from generalized_hill_climbing import generalized_hill_climbing
from creer_le_pickle import creer_pickle
import pandas as pd


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
    penalite = 75
    fonction = fonction_objectives(valeurs_in)

    if verification_contraintes(valeurs_in) == False :
        penalite += penalite *1.1
        return fonction + penalite
    else :
        return fonction


def analyser_pickle(fichier=r"C:\Users\delha\OneDrive\Desktop\Cours_UQAR\Metaheuristique\resultats.pkl", algo="hc"):
    df = pd.read_pickle(fichier)

    data = df[["algo", "itération", "coût"]].groupby(
        ["algo", "itération"], as_index=False
    ).agg(
        min=pd.NamedAgg(column="coût", aggfunc=np.min),
        max=pd.NamedAgg(column="coût", aggfunc="max"),
        médiane=pd.NamedAgg(column="coût", aggfunc="median"),
        q1=pd.NamedAgg(column="coût", aggfunc=lambda x: np.quantile(x, 0.25)),
        q3=pd.NamedAgg(column="coût", aggfunc=lambda x: np.quantile(x, 0.75)),
    )

    data_algo = data[data["algo"] == algo]

    ax = data_algo.plot(x="itération", y="médiane", label=f"{algo}_médiane")
    ax.plot(data_algo["q1"], alpha=0.5)
    ax.plot(data_algo["q3"], alpha=0.5)

    ax.fill_between(
        x=data_algo["itération"],
        y1=data_algo["min"],
        y2=data_algo["max"],
        alpha=0.2,
    )

    ax.set_yscale("log")
    plt.legend()
    plt.show()
        

def main():
    n=50
    variable = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]

    algo_recherche_aleatoire = RandomSearch(n, penaliser_algo)
    meilleur_x, meilleur_solution, history = algo_recherche_aleatoire.run()
    print("Meilleure solution : ", meilleur_x)
    print("Meilleure valeur : ", meilleur_solution)


    algo_hill_climbing1_1 = Hill_Climbing_1_1(n, penaliser_algo, variable, 0.05)
    meilleur_x_hc, meilleur_solution_hc, history_hc = algo_hill_climbing1_1.run()

    
    print("Meilleure solution Hill Climbing : ", meilleur_x_hc)
    print("Meilleure valeur Hill Climbing : ", meilleur_solution_hc)

    algo_hill_climbing = generalized_hill_climbing(n, penaliser_algo, variable, 0.05, 8)
    meilleur_x_hc_G, meilleur_solution_hc_G, history_hc_G = algo_hill_climbing.run()

    print("Meilleure solution Hill Climbing : ", meilleur_x_hc_G)
    print("Meilleure valeur Hill Climbing : ", meilleur_solution_hc_G)

    plt.plot( [h[1] for h in history], label='Recherche Aléatoire')
    plt.show()
    plt.plot( [h[1] for h in history_hc], label='Hill Climbing')
    plt.show()
    plt.plot( [h[1] for h in history_hc_G], label='Hill Climbing Généralisé')
    plt.title('Comparaison des algorithmes')
    plt.legend()
    plt.show()

    """
    df = creer_pickle(
    penaliser_algo=penaliser_algo,
    variable=variable,
    n_iter=n,
    nb_simulations=100,
    pas=0.05,
    nbr_voisin=8,
    outfile="resultats.pkl",
    outcsv="resultats.csv",
    )
    print("Pickle créé:", "resultats.pkl")"""

    analyser_pickle(fichier=r"C:\Users\delha\OneDrive\Desktop\Cours_UQAR\Metaheuristique\resultats.pkl", algo="rand")


if __name__ == "__main__":    
    main()
    