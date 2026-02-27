import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulated_annealing import SimulatedAnnealing

def load_results(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        raise ValueError("Fichier doit être .csv ou .pkl")

    
    required = {"algo", "run_no", "iteration", "cout"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes attendues: {required}, trouvé: {set(df.columns)}")

    df["run_no"] = df["run_no"].astype(int)
    df["iteration"] = df["iteration"].astype(int)
    df["cout"] = df["cout"].astype(float)
    return df

def stats_finales(df):

    df_final = (df.sort_values(["algo", "run_no", "iteration"]).groupby(["algo", "run_no"]).tail(1).reset_index(drop=True))

    stats = df_final.groupby("algo")["cout"].agg(best="min",median="median",std="std").reset_index()

    #df_final = df.sort_values(["algo", "run_no", "iteration"]).groupby(["algo", "run_no"])


    q = df_final.groupby("algo")["cout"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    stats = stats.merge(q, on="algo")
    stats["IQR"] = stats["q3"] - stats["q1"]

    return stats, df_final


def convergence(df):
    agg = df.groupby(["algo", "iteration"])["cout"].agg(median="median",q1=lambda x: np.quantile(x, 0.25),q3=lambda x: np.quantile(x, 0.75)).reset_index()
    return agg


def plot_convergence(agg, use_log=True, title="Convergence (médiane + IQR)"):
    plt.figure(figsize=(11, 6))

    for algo in agg["algo"].unique():
        d = agg[agg["algo"] == algo].sort_values("iteration")
        x = d["iteration"].to_numpy()
        y = d["median"].to_numpy()
        q1 = d["q1"].to_numpy()
        q3 = d["q3"].to_numpy()

        plt.plot(x, y, label=algo)
        plt.fill_between(x, q1, q3, alpha=0.2)

    plt.xlabel("Itération")
    plt.ylabel("Coût")
    plt.title(title)
    plt.legend()
    if use_log:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()

def plot_boxplot_final(df_final, use_log=True, title="coûts finaux de Monte-Carlo"):
    algos = sorted(df_final["algo"].unique())
    data = [df_final[df_final["algo"] == a]["cout"].to_numpy() for a in algos]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=algos, showfliers=True)
    if use_log:
        plt.yscale("log")
    plt.xlabel("Algorithme")
    plt.ylabel("Coût final")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def write_protocol(filename, nb_simulations, n_iter, pas, nbr_voisin, refroidissement, epsilon, patience):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Protocole expérimental — TP01 8INF852\n\n")
        f.write(f"Monte-Carlo: {nb_simulations} runs par algorithme\n")
        f.write(f"Itérations max: {n_iter}\n")
        f.write(f"Pas (voisinage): {pas}\n")
        f.write(f"Voisins GHC (lambda): {nbr_voisin}\n")
        f.write(f"Refroidissement SA: {refroidissement}\n")
        f.write(f"Stagnation: epsilon={epsilon}, patience={patience}\n")
        

def run_metrics(results_path, nb_simulations, n_iter, pas, nbr_voisin, refroidissement, epsilon, patience):
    df = load_results(results_path)

    stats, df_final = stats_finales(df)
    part1, part2 = results_path.split("resultats.csv")

    write_protocol(part1 + r"\protocole.txt", nb_simulations, n_iter, pas, nbr_voisin, refroidissement, epsilon, patience)

    agg = convergence(df)
    plot_convergence(agg, use_log=True)
    plot_boxplot_final(df_final, use_log=True)

    return stats


def compare_refroidissements(fonction, variable, runs=30, n_iter=1000):
    

    modes = ["lineaire", "exponentielle", "logarithmique", "geometrique"]
    all_results = []

    for mode in modes:
        for r in range(runs):
            sa = SimulatedAnnealing(n_iter=n_iter,fonction=fonction,variable=variable,pas=0.05,refroidissement=mode)

            _, _, history = sa.run()

            for (it, cost) in history:all_results.append({"mode": mode,"run": r,"iteration": it,"cost": cost})

    df = pd.DataFrame(all_results)

    agg = df.groupby(["mode", "iteration"])["cost"].median().reset_index()

    plt.figure(figsize=(10,6))
    for mode in modes:
        d = agg[agg["mode"] == mode]
        plt.plot(d["iteration"], d["cost"], label=mode)

    plt.yscale("log")
    plt.xlabel("Itération")
    plt.ylabel("Coût médian")
    plt.title("Comparaison des refroidissements (SA)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    df_final = (df.sort_values(["mode", "run", "iteration"]).groupby(["mode", "run"]).tail(1))

    print("\nMédiane des coûts finaux :")
    print(df_final.groupby("mode")["cost"].median())

    return df