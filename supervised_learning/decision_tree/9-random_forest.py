#!/usr/bin/env python3
"""
Təsadüfi Meşə (Random Forest) ansambl modelini təmsil edən modul
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Qərar ağacları qrupundan ibarət Təsadüfi Meşə sinfi"""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Random Forest obyektinin parametrlərini başladır"""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Bütün ağacların proqnozları əsasında çoxluq səsverməsi ilə

        ən optimal sinfi təyin edir
        """
        # Hər bir ağacdan fərdlər üzrə proqnozları toplayırıq
        tree_predictions = np.array([pred(explanatory)
                                     for pred in self.numpy_preds])

        # Hər bir fərd üçün ən çox təkrarlanan (mode) sinfi hesablayırıq
        num_individuals = explanatory.shape[0]
        modes = np.array([np.bincount(tree_predictions[:, i]).argmax()
                          for i in range(num_individuals)])

        return modes

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Təsadüfi bölgü kriteriyası ilə ağaclar quraraq meşəni öyrədir"""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print("  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")
            print(f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """Meşə modelinin test verilənləri üzərində dəqiqliyini hesablayır"""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target
        )) / test_target.size
