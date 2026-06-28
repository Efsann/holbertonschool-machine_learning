#!/usr/bin/env python3
"""
Anomal dəyərləri tapmaq üçün İzolyasiya Ağacı moduludur
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Anomallıq analizi üçün təsadüfi izolyasiya ağacı sinfi"""

    def __init__(self, max_depth=10, seed=0, root=None):
        """İzolyasiya ağacının ilkin parametrlərini başladır"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """Ağac strukturunu sətir kimi qaytarır"""
        return self.root.__str__()

    def depth(self):
        """Ağacın maksimum dərinliyini qaytarır"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Ağacdakı düyün və ya yarpaqların sayını hesablayır"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Bütün düyünlərin həndəsi sərhədlərini yeniləyir"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Ağacın bütün son yarpaqlarını çəkir"""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Sürətli proqnoz funksiyasını yaradır"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], axis=0
        )

    def np_extrema(self, arr):
        """Massivin minimum və maksimum dəyərlərini qaytarır"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Təsadüfi olaraq əlamət və sərhəd həddi seçir"""
        sub_X = self.explanatory[node.sub_population]
        if np.all(sub_X == sub_X[0]):
            return 0, 0.0

        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            values = self.explanatory[:, feature][
                node.sub_population
            ]
            feature_min, feature_max = self.np_extrema(values)
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Yarpaq obyekti yaradır və
        dəyər olaraq onun dərinliyini təyin edir.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Yeni bir daxili düyün obyekti yaradır"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Düyünləri ancaq populyasiya və dərinliyə görə bölür"""
        sub_X = self.explanatory[node.sub_population]
        if len(sub_X) <= self.min_pop or np.all(sub_X == sub_X[0]):
            return

        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & ~ (
            self.explanatory[:, node.feature] > node.threshold
        )

        is_left_leaf = (
            np.sum(left_population) < self.min_pop or
            node.depth + 1 == self.max_depth or
            np.all(self.explanatory[left_population] ==
                   self.explanatory[left_population][0])
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) < self.min_pop or
            node.depth + 1 == self.max_depth or
            np.all(self.explanatory[right_population] ==
                   self.explanatory[right_population][0])
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """İzolyasiya ağacını verilən massiv üzərində öyrədir"""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
