import numpy as np

class SimulatedAnnealing:
    def __init__(self, n_iter, fonction, variable, pas, refroidissement, temperature_initiale=0.01):
        self.n_iter = n_iter
        self.fonction = fonction
        self.variable = variable
        self.pas = pas
        self.refroidissement = refroidissement
        self.temperature_initiale = float(temperature_initiale)

    def creation_solution_initiale(self):
        return np.array([np.random.uniform(a, b) for (a, b) in self.variable])

    def choix_voisin(self, x):
        voisin = np.copy(x)
        variable_hasard = np.random.choice(len(x))

        if np.random.rand() < 0.5:
            voisin[variable_hasard] += self.pas
        else:
            voisin[variable_hasard] -= self.pas

        for j in range(len(voisin)):
            if voisin[j] < self.variable[j][0]:
                voisin[j] = self.variable[j][0]
            elif voisin[j] > self.variable[j][1]:
                voisin[j] = self.variable[j][1]
        return voisin

    def refroidissement_temp(self, temp, iteration):
        mode = self.refroidissement.lower()
        tmin = 0.001

        if mode == "lineaire":
            return max(tmin, temp - (self.temperature_initiale / self.n_iter))

        elif mode == "exponentielle":
            return max(tmin, temp * (1.0 - 1.0 / self.n_iter))

        elif mode == "logarithmique":
            return max(tmin, self.temperature_initiale / np.log(iteration + 2))

        elif mode == "geometrique":
            return max(tmin, temp * (1.0 - 2.0 / self.n_iter))

        else:
            raise ValueError(f"Mode inconnu : {self.refroidissement}")

    def rechauffement(self, best_x, best_fx):
        x = best_x.copy()
        fx = float(best_fx)
        temp = self.temperature_initiale
        return x, fx, temp

    def run(self, epsilon=0.001, patience=50):
        x = self.creation_solution_initiale()
        fx = float(self.fonction(x))

        best_x = x.copy()
        best_fx = fx

        temp = self.temperature_initiale  

        history = [(0, fx)]
        stagn = 0

        for i in range(1, self.n_iter + 1):

            voisin = self.choix_voisin(x)
            f_voisin = float(self.fonction(voisin))

            if f_voisin < fx:
                x, fx = voisin, f_voisin
            else:
                proba = np.exp((fx - f_voisin) / max(temp, 1e-12))
                if np.random.rand() < proba:
                    x, fx = voisin, f_voisin

            if fx < best_fx - epsilon:
                best_x = x.copy()
                best_fx = fx
                stagn = 0
            else:
                stagn += 1

            history.append((i, fx))

            temp = self.refroidissement_temp(temp, i)

            if stagn >= patience:
                x, fx, temp = self.rechauffement(best_x, best_fx)
                stagn = 0
                history.append((i, fx))  

        return best_x, best_fx, history