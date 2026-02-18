import pandas as pd
from random_search import RandomSearch
from Hill_Climbing_simple_1_1 import Hill_Climbing_1_1
from generalized_hill_climbing import generalized_hill_climbing


def creer_pickle(
    penaliser_algo,
    variable,
    n_iter=50,
    nb_simulations=100,
    pas=0.05,
    nbr_voisin=8,
    outfile="resultats.pkl",
    outcsv="resultats.csv",
):
 
    colonnes = ["algo", "run_no", "itération", "coût"]
    rows = []

    for s in range(nb_simulations):
        rs_algo = RandomSearch(n_iter, penaliser_algo)
        _, _, hist_rs = rs_algo.run()  
        for it, x in hist_rs:
            if x is None:
                continue
            rows.append(("rand", s, int(it), float(penaliser_algo(x))))

        hc_algo = Hill_Climbing_1_1(n_iter, penaliser_algo, variable, pas)
        _, _, hist_hc = hc_algo.run() 
        for it, x in hist_hc:
            rows.append(("hc", s, int(it), float(penaliser_algo(x))))

        ghc_algo = generalized_hill_climbing(n_iter, penaliser_algo, variable, pas, nbr_voisin)
        _, _, hist_ghc = ghc_algo.run() 
        for it, x in hist_ghc:
            rows.append(("ghc", s, int(it), float(penaliser_algo(x))))

    df = pd.DataFrame(rows, columns=colonnes)
    df.to_pickle(outfile)
    if outcsv:
        df.to_csv(outcsv, index=False)

    return df
