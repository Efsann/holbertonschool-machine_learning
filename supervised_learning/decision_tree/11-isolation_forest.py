#!/usr/bin/env python3
"""
Anomal dəyərlərin aşkarlanması üçün Təsadüfi İzolyasiya Meşəsi modulu
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """Ansambl izolyasiya ağaclarından ibarət anomal aşkarlama modeli"""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """İzolyasiya meşəsinin başlanğıc parametrlərini qurur"""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """Bütün ağaclar üzrə fərdlərin ortalama təcrid dərinliyini tapır"""
        predictions = np.array([f(explanatory)
                                for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """Meşə daxilində n sayda fərdi izolyasiya ağacını öyrədir"""
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print("  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")

    def suspects(self, explanatory, n_suspects):
        """Ortalama dərinliyi ən kiçik olan n şübhəli anomal sətiri tapır"""
        depths = self.predict(explanatory)
        indices = np.argsort(depths)[:n_suspects]
        return explanatory[indices], depths[indices]
