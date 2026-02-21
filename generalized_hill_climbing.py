import numpy as np

class generalized_hill_climbing :
    def __init__ (self, n_iter, fonction, variable, pas, nbr_voisin):
        self.n_iter = n_iter
        self.fonction = fonction
        self.variable = variable
        self.pas = pas
        self.nbr_voisin = nbr_voisin
    
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
    
    def stagnation_check(self, history, stagnation_threshold=20, epsilon=0.001):

        if len(history) < stagnation_threshold:
            return False

        last_costs = [h[1] for h in history[-stagnation_threshold:]]

        improvement = abs(last_costs[0] - last_costs[-1])

        return improvement < epsilon

        
    def run(self):
        x = self.creation_solution_initiale()
        p = float(self.fonction(x))

        best_x = x.copy()
        best_p = p

        stagn = 0
        history = [(0, p)]

        for i in range(1, self.n_iter + 1):

            
            best_voisin = None
            best_voisin_p = p

            for _ in range(self.nbr_voisin):
                voisin = self.choix_voisin(x)
                pv = float(self.fonction(voisin))

                if pv < best_voisin_p:
                    best_voisin_p = pv
                    best_voisin = voisin

           
            if best_voisin is not None and best_voisin_p < p:
                improvement = p - best_voisin_p
                x = best_voisin
                p = best_voisin_p

                if improvement > 0.001:  
                    stagn = 0
                else:
                    stagn += 1
            else:
                stagn += 1

            
            if p < best_p:
                best_p = p
                best_x = x.copy()

            history.append((i, best_p))
            #history.append((i, p))

            
            if stagn >= 50:  
                x = self.creation_solution_initiale()
                p = float(self.fonction(x))
                stagn = 0

        return best_x, best_p, history