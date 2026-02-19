import numpy as np

class Hill_Climbing_1_1:

    def __init__ (self, n_iter, fonction, variable, pas):
        self.n_iter = n_iter
        self.fonction = fonction
        self.variable = variable
        self.pas = pas
        #np.random.seed(42)
    
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
 

    def creation_solution_initiale(self):
        x = []
        for var in self.variable:
            x.append(np.random.uniform(var[0], var[1]))
        return np.array(x)
    
    def run(self):
        x =self.creation_solution_initiale()
        fonction = self.fonction(x)

        history = [(0, x.copy())]

        for i in range(self.n_iter):
            voisin = self.choix_voisin(x)
            f_voisin = self.fonction(voisin)

            if f_voisin < fonction:
                x = voisin
                fonction = f_voisin
            history.append((i, fonction))

        return x, fonction, history




