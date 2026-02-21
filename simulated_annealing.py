import numpy as np

class SimulatedAnnealing:
    def __init__ (self, n_iter, fonction, variable, pas, refroidissement):
        self.n_iter = n_iter
        self.fonction = fonction
        self.variable = variable
        self.pas = pas
        self.refroidissement = refroidissement

    def creation_solution_initiale(self):
        x = []
        for var in self.variable:
            x.append(np.random.uniform(var[0], var[1]))
        return np.array(x)
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
    
    def refroidissement_temp(self, iteration):
        mode = self.refroidissement.lower()

        if mode == "lineaire":
            return max(0.01, 1 - iteration / self.n_iter)

        elif mode == "exponentielle":
            return max(0.01, 0.95 ** iteration)

        elif mode == "logarithmique":
            return max(0.01, 1 / np.log(iteration + 2))

        elif mode == "geometrique":
            return max(0.01, 0.9 ** iteration)

        else:
            raise ValueError(f"Mode inconnu : {self.refroidissement}")
        
    def rechauffement_temp(self, best_x, best_fx):
        x = best_x.copy()
        fx = float(best_fx)
        return x, fx

    def stagnation_check(self, history, stagnation_threshold=50, epsilon=0.001):

        if len(history) < stagnation_threshold:
            return False
        
        last_costs = [h[1] for h in history[-stagnation_threshold:]]

        improvement = abs(last_costs[0] - last_costs[-1])

        return improvement < epsilon

    
    def run(self):
        x = self.creation_solution_initiale()
        fonction = float(self.fonction(x))

        best_x = x.copy()
        best_fx = fonction

        history = [(0, fonction)]  

        for i in range(1, self.n_iter + 1):

            voisin = self.choix_voisin(x)
            f_voisin = float(self.fonction(voisin))

            if f_voisin < fonction:
                x = voisin
                fonction = f_voisin
            else:
                temp = self.refroidissement_temp(i)
                proba = np.exp((fonction - f_voisin) / max(temp, 1e-12))
                if np.random.rand() < proba:
                    x = voisin
                    fonction = f_voisin

            if fonction < best_fx:
                best_x = x.copy()
                best_fx = fonction

            history.append((i, fonction))

            if self.stagnation_check(history, stagnation_threshold=50):
                x, fonction = self.rechauffement_temp(best_x, best_fx)

        return best_x, best_fx, history

        
 