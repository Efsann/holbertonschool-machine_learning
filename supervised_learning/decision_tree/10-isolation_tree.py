#!/usr/bin/env python3
"""
Isolation Random Tree modulunun reallaşdırılması
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Anomaliyaları aşkar etmək üçün Isolation Random Tree sinfi"""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Obyektin ilkin parametrlərini başladır"""
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
        """Ağacın strukturunu sətir kimi vizuallaşdırır"""
        return self.root.__str__()

    def depth(self):
        """Ağacın maksimum dərinliyini qaytarır"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Ağacdakı düyünlərin və ya yarpaqların sayını hesablayır"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Hər bir düyün üçün sərhədləri yeniləyir"""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Ağacdakı bütün yarpaqların siyahısını qaytarır"""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Proqnozlaşdırma funksiyasını yeniləyir"""
        self.root.update_predict_below()

    def np_extrema(self, arr):
        """Massivin minimum və maksimum dəyərlərini qaytarır"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Təsadüfi bölmə kriteriyasını hesablayır"""
        return node.random_split_criterion()

    def get_leaf_child(self, node, sub_population):
        """Yarpaq övlad obyektini yaradır və nizamlayır"""
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Daxili düyün övlad obyektini yaradır və nizamlayır"""
        return node.get_node_child(sub_population)

    def fit_node(self, node):
        """Düyünü təsadüfi kriteriyalara əsasən bölür"""
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.left_child_sub_population()
        right_population = node.right_child_sub_population()

        # Sol düyünün yarpaq olub-olmaması yoxlanılır
        is_left_leaf = (node.depth + 1 >= self.max_depth or
                        np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Sağ düyünün yarpaq olub-olmaması yoxlanılır
        is_right_leaf = (node.depth + 1 >= self.max_depth or
                         np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Verilən məlumatlar əsasında Isolation Tree modelini öyrədir"""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")
