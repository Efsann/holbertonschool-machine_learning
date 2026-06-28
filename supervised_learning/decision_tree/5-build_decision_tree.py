#!/usr/bin/env python3
"""
Qərar ağacını çap etmək və idarə etmək üçün vizuallaşdırma moduludur
"""
import numpy as np


class Node:
    """Qərar ağacındakı daxili düyünü təmsil edən sinif"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Düyündən aşağıda qalan maksimum dərinliyi hesablayır"""
        left_depth = (self.left_child.max_depth_below()
                      if self.left_child else self.depth)
        right_depth = (self.right_child.max_depth_below()
                       if self.right_child else self.depth)
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Düyündən aşağıda qalan düyün və ya yarpaqların sayını hesablayır"""
        left_count = (self.left_child.count_nodes_below(
            only_leaves=only_leaves) if self.left_child else 0)
        right_count = (self.right_child.count_nodes_below(
            only_leaves=only_leaves) if self.right_child else 0)

        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """Sol övladın budaqları üçün prefiks əlavə edir"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0]
        for x in lines[1:]:
            new_text += "\n    |  " + x
        return new_text

    def right_child_add_prefix(self, text):
        """Sağ övladın budaqları üçün prefiks əlavə edir"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0]
        for x in lines[1:]:
            new_text += "\n       " + x
        return new_text

    def get_leaves_below(self):
        """Düyündən aşağıda qalan bütün yarpaqların siyahısını qaytarır"""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Düyünlərin alt sərhədlərini rekursiv hesablayır"""
        if self.is_root:
            self.upper = {}
            self.lower = {}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold
            self.left_child.update_bounds_below()

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold
            self.right_child.update_bounds_below()

    def update_indicator(self):
        """Hesablanmış sərhədlərə əsasən indikator funksiyasını yeniləyir"""

        def is_large_enough(x):
            if not self.lower:
                return np.ones(x.shape[0], dtype=bool)
            return np.all(np.array(
                [np.greater(x[:, k], self.lower[k]) for k in self.lower.keys()]
            ), axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            return np.all(np.array(
                [np.less_equal(x[:, k], self.upper[k]) for k in self.upper.keys()]
            ), axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]
        ), axis=0)

    def __str__(self):
        """Düyünü sətir formatına çevirir"""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child:
            out += "\n" + self.left_child_add_prefix(
                self.left_child.__str__())
        if self.right_child:
            out += "\n" + self.right_child_add_prefix(
                self.right_child.__str__())

        return out


class Leaf(Node):
    """Qərar ağacındakı yarpaq düyünü təmsil edən sinif"""

    def __init__(self, value, depth=0):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def get_leaves_below(self):
        return [self]

    def update_bounds_below(self):
        pass

    def __str__(self):
        return f"-> leaf [value={self.value}]"


class Decision_Tree():
    """Qərar ağacı modelini idarə edən əsas sinif"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_bounds(self):
        self.root.update_bounds_below()

    def __str__(self):
        return self.root.__str__()
