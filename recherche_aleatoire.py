import numpy as np

class RechercheAleatoire:

    def __init__ (self, n_iter, fonction):
        self.n_iter = n_iter
        self.fonction = fonction
        

    
    def run(self):
        meilleur_x = None
        meilleur_solution = float('inf')
        history = []


        for i in range(self.n_iter):
            x1 = np.random.uniform(0.05, 2.0)
            x2 = np.random.uniform(0.25, 1.3)
            x3 = np.random.uniform(2.0, 15.0)

            valeurs_test = (x1, x2, x3)

            if self.fonction(valeurs_test) < meilleur_solution:
                meilleur_solution = self.fonction(valeurs_test)
                meilleur_x = valeurs_test
            history.append((i, meilleur_x))

        return meilleur_x, meilleur_solution, history