#!/usr/bin/env python3
"""
Qərar ağacını qurmaq və vizuallaşdırmaq üçün modul
"""
import numpy as np


class Node:
    """Qərar ağacındakı daxili düyünü təmsil edən sinif"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Düyün obyektinin ilkin parametrlərini başladır"""
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
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Sağ övladın budaqları üçün prefiks əlavə edir"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Düyünü sətir formatına çevirir"""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"-> node [feature={self.feature}, " \
                  f"threshold={self.threshold}]\n"

        if self.left_child:
            out += self.left_child_add_prefix(
                self.left_child.__str__().rstrip("\n"))
        if self.right_child:
            out += self.right_child_add_prefix(
                self.right_child.__str__().rstrip("\n"))

        return out


class Leaf(Node):
    """Qərar ağacındakı yarpaq düyünü təmsil edən sinif"""

    def __init__(self, value, depth=0):
        """Yarpaq obyektinin parametrlərini başladır"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Yarpağın dərinliyini qaytarır"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Yarpaq sayını qaytarır (həmişə 1)"""
        return 1

    def __str__(self):
        """Yarpağı sətir formatına çevirir"""
        return f"-> leaf [value={self.value}]"


class Decision_Tree():
    """Qərar ağacı modelini idarə edən əsas sinif"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Qərar ağacı obyektini başladır"""
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
        """Ağacın dərinliyini hesablayır"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Ağacın düyünlərini hesablayır"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Ağacı bütövlükdə sətir kimi vizuallaşdırır"""
        return self.root.__str__()
