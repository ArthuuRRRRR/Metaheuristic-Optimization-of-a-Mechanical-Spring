import numpy as np

class Hill_Climbing:
    
    def __init__ (self, n_iter, fonction, variable, pas):
        self.n_iter = n_iter
        self.fonction = fonction
        self.variable = variable
        self.pas = pas
    
    def choix_voisin(self, x):
        pass

    def creation_solution_initiale(self):
        x = []
        for var in self.variable:
            x.append(np.random.uniform(var[0], var[1]))
        return np.array(x)
    
    def run(self):
        x =self.creation_solution_initiale()
        fonction = self.contraintes_fonction(x)

        history = []

        for i in range(self.n_iter):
            voisin = self.choix_voisin(x)
            f_voisin = self.contraintes_fonction(voisin)

            if f_voisin < fonction:
                x = voisin
                fonction = f_voisin
            history.append((i, fonction))

        return x, fonction, history





