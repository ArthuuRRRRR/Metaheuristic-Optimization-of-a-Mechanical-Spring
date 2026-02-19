import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class display_result:
    def load_results(path=r"C:\Users\delha\OneDrive\Desktop\Cours_UQAR\Metaheuristique\resultats.csv"):
        if path.endswith(".pkl"):
            df = pd.read_pickle(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("Le fichier doit être .pkl ou .csv")

        required = {"algo", "run_no", "iteration", "cout"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}. Colonnes trouvées: {list(df.columns)}")

        df = df.copy()
        df["iteration"] = df["iteration"].astype(int)
        df["cout"] = df["cout"].astype(float)

        return df

    def load_results(path=r"C:\Users\delha\OneDrive\Desktop\Cours_UQAR\Metaheuristique\resultats.csv"):
        if path.endswith(".pkl"):
            df = pd.read_pickle(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("Le fichier doit être .pkl ou .csv")


        required = {"algo", "run_no", "iteration", "cout"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}. Colonnes trouvées: {list(df.columns)}")

        df = df.copy()
        df["iteration"] = df["iteration"].astype(int)
        df["cout"] = df["cout"].astype(float)

        return df


    def aggregate_convergence(df: pd.DataFrame) -> pd.DataFrame:
        agg = (
            df.groupby(["algo", "iteration"])["cout"]
            .agg(
                mediane="median",
                q1=lambda x: np.quantile(x, 0.25),
                q3=lambda x: np.quantile(x, 0.75),
                min_="min",
                max_="max",
            )
            .reset_index()
            .sort_values(["algo", "iteration"])
        )
        return agg


    def plot_all_algos(
        agg: pd.DataFrame,
        algos_order=("RS", "HC", "GHC", "SA"),
        use_log=True,
        show_minmax=False,
        title="Profils de convergence (médiane + IQR)",
    ):
        plt.figure(figsize=(11, 6))

        present = list(agg["algo"].unique())
        order = [a for a in algos_order if a in present] + [a for a in present if a not in algos_order]

        for algo in order:
            d = agg[agg["algo"] == algo].sort_values("iteration")

            x = d["iteration"].to_numpy()
            y = d["mediane"].to_numpy()
            q1 = d["q1"].to_numpy()
            q3 = d["q3"].to_numpy()

            plt.plot(x, y, label=f"{algo} médiane")

            plt.fill_between(x, q1, q3, alpha=0.20)

            if show_minmax:
                plt.fill_between(x, d["min_"].to_numpy(), d["max_"].to_numpy(), alpha=0.08)

        plt.xlabel("Itération")
        plt.ylabel("Coût")
        plt.title(title)
        plt.legend()

        if use_log:
            plt.yscale("log")

        plt.tight_layout()
        plt.show()
