import numpy as np

class HillClimbing:
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
        variable_hasard = np.random.choice([0, 1, 2])

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
    
    def temperature(self, iteration):
        pass
    
    def run(self):
        pass
        
 