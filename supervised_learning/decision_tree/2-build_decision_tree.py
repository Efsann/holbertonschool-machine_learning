#!/usr/bin/env python3
"""
Qərar ağacının vizuallaşdırılması üçün __str__ metodlarının reallaşdırılması
"""
import numpy as np


class Node:
    """Qərar ağacının daxili düyünlərini təmsil edən sinif"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, depth=0, is_root=False):
        """Düyün obyektinin ilkin parametrlərini başladır"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def left_child_add_prefix(self, text):
        """Sol budaqdan gələn alt sətirlərin başına iyerarxik xətlər əlavə edir"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Sağ budaqdan gələn alt sətirlərin başına boşluqlar əlavə edir"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            # Əgər sətir boşdursa, sonuna lazımsız boşluq əlavə etmirik
            if x:
                new_text += ("       " + x) + "\n"
            else:
                new_text += "\n"
        return new_text

    def __str__(self):
        """Düyünü və onun alt budaqlarını strukturlaşdırılmış mətnə çevirir"""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]\n"
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]\n"

        if self.left_child:
            left_str = self.left_child.__str__()
            out += self.left_child_add_prefix(left_str)

        if self.right_child:
            right_str = self.right_child.__str__()
            out += self.right_child_add_prefix(right_str)

        # Çap formatının main daxilindəki çıxışla tam uzlaşması üçün 
        # tənzimləmə edirik və artıq n-ləri silirik
        return out.strip("\n")


class Leaf:
    """Qərar ağacının son yarpaqlarını təmsil edən sinif"""

    def __init__(self, value, depth=0):
        """Yarpaq obyektinin ilkin parametrlərini başladır"""
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def __str__(self):
        """Yarpağın terminal formatında mətn qarşılığını qaytarır"""
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """Qərar Ağacı modelinin əsas idarəedici sinfi"""

    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="gini",
                 root=None):
        """Qərar ağacı strukturunu başladır"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        """Ağacı kökündən başlayaraq çap edir"""
        return self.root.__str__()
