import pandas as pd
import numpy as np

from random_search import RandomSearch
from Hill_Climbing_simple_1_1 import Hill_Climbing_1_1
from generalized_hill_climbing import generalized_hill_climbing
from simulated_annealing import SimulatedAnnealing


def creer_pickle(
    penaliser_algo,
    variable,
    n_iter=200,
    nb_simulations=50,
    pas=0.05,
    nbr_voisin=6,
    refroidissement="exponentielle",
    outfile="resultats7.pkl",
    outcsv="resultats7.csv",
):

    colonnes = ["algo", "run_no", "iteration", "cout"]
    rows = []

    def to_cost(value):

        if np.isscalar(value):
            return float(value)
        else:
            return float(penaliser_algo(value))

    for run in range(nb_simulations):

        print(f"Simulation {run+1}/{nb_simulations}")

        rs = RandomSearch(n_iter, penaliser_algo)
        _, _, hist = rs.run()

        for it, val in hist:
            rows.append(("RS", run, int(it), to_cost(val)))

        hc = Hill_Climbing_1_1(n_iter, penaliser_algo, variable, pas)
        _, _, hist = hc.run()

        for it, val in hist:
            rows.append(("HC", run, int(it), to_cost(val)))

        ghc = generalized_hill_climbing(
            n_iter, penaliser_algo, variable, pas, nbr_voisin
        )
        _, _, hist = ghc.run()

        for it, val in hist:
            rows.append(("GHC", run, int(it), to_cost(val)))

        sa = SimulatedAnnealing(
            n_iter, penaliser_algo, variable, pas, refroidissement
        )
        _, _, hist = sa.run()

        for it, val in hist:
            rows.append(("SA", run, int(it), to_cost(val)))

    df = pd.DataFrame(rows, columns=colonnes)

    df.to_pickle(outfile)

    if outcsv:
        df.to_csv(outcsv, index=False)

    print("\nPickle créé :", outfile)
    print("CSV créé :", outcsv)

    return df
