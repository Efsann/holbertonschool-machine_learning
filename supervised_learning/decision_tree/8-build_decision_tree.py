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
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

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
                [np.greater(x[:, k], self.lower[k])
                 for k in self.lower.keys()]
            ), axis=0)

        def is_small_enough(x):
            if not self.upper:
                return np.ones(x.shape[0], dtype=bool)
            return np.all(np.array(
                [np.less_equal(x[:, k], self.upper[k])
                 for k in self.upper.keys()]
            ), axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]
        ), axis=0)

    def pred(self, x):
        """Fərdi sətir üzrə ağacı gəzərək proqnoz verir"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

    def __str__(self):
        """Düyünü sətir formatına çevirir"""
        if self.is_root:
            out = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            out = f"node [feature={self.feature}, threshold={self.threshold}]"

        if self.left_child:
            out += "\n" + self.left_child_add_prefix(self.left_child.__str__())
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
        """Yarpağın öz dərinliyini qaytarır"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Yarpaq hər zaman 1 olaraq sayılır"""
        return 1

    def get_leaves_below(self):
        """Yarpaq özü bir yarpaq olduğu üçün özünü siyahıda qaytarır"""
        return [self]

    def update_bounds_below(self):
        """Yarpaq son nöqtə olduğu üçün sərhəd yeniləməsini dayandırır"""
        pass

    def pred(self, x):
        """Yarpağa çatdıqda birbaşa hədəf dəyəri qaytarır"""
        return self.value

    def __str__(self):
        """Yarpağı sətir formatına çevirir"""
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
        """Ağacın ümumi maksimum dərinliyini qaytarır"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Ağacda olan düyünlərin sayını hesablayır"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """Ağacın kökündən başlayaraq bütün yarpaqları çəkir"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Bütün ağac üzrə sərhədlərin hesablanmasını başladır"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Kütləvi matris əməliyyatları üçün proqnoz funksiyasını qurur"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.indicator(A) * leaf.value for leaf in leaves], axis=0
        )

    def pred(self, x):
        """Kökdən başlayaraq fərdi proqnoz funksiyasını başladır"""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Massivin minimum və maksimum dəyərlərini qaytarır"""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Təsadüfi olaraq əlamət və sərhəd həddi seçir"""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def possible_thresholds(self, node, feature):
        """Müvafiq əlamət üçün bütün unikal orta bölgü hədlərini tapır"""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Bir əlamət üzrə ən optimal Gini kəsimini və həddini hesablayır"""
        thresholds = self.possible_thresholds(node, feature)
        if len(thresholds) == 0:
            return 0.0, np.inf

        X_feat = self.explanatory[:, feature][node.sub_population]
        Y_node = self.target[node.sub_population]
        classes = np.unique(self.target)

        # 3D yayım matrisinin (n, t, c) qurulması
        X_expanded = X_feat[:, np.newaxis, np.newaxis]
        T_expanded = thresholds[np.newaxis, :, np.newaxis]
        C_expanded = classes[np.newaxis, np.newaxis, :]
        Y_expanded = Y_node[:, np.newaxis, np.newaxis]

        Left_F = (Y_expanded == C_expanded) & (X_expanded > T_expanded)
        Right_F = (Y_expanded == C_expanded) & (X_expanded <= T_expanded)

        left_counts = Left_F.sum(axis=0)
        right_counts = Right_F.sum(axis=0)

        left_ns = left_counts.sum(axis=1)
        right_ns = right_counts.sum(axis=1)
        n_total = len(X_feat)

        # Sıfıra bölünmə xətasının qarşısını almaq üçün maskalama
        left_ns_mask = np.where(left_ns == 0, 1, left_ns)
        right_ns_mask = np.where(right_ns == 0, 1, right_ns)

        gini_left = 1 - np.sum((left_counts / left_ns_mask[:, None]) ** 2,
                               axis=1)
        gini_right = 1 - np.sum((right_counts / right_ns_mask[:, None]) ** 2,
                                axis=1)

        gini_left = np.where(left_ns == 0, 0.0, gini_left)
        gini_right = np.where(right_ns == 0, 0.0, gini_right)

        gini_total = (left_ns / n_total) * gini_left + (
            right_ns / n_total) * gini_right

        best_idx = np.argmin(gini_total)
        return thresholds[best_idx], gini_total[best_idx]

    def Gini_split_criterion(self, node):
        """Bütün əlamətləri gəzərək minimum ümumi Gini çirkliliyini tapır"""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def fit(self, explanatory, target, verbose=0):
        """Qərar ağacını seçilmiş kriteriyaya uyğun öyrədir"""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print("  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def fit_node(self, node):
        """Düyünləri rekursiv şəkildə böyüdərək ağacı qurur"""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = node.sub_population & ~ (
            self.explanatory[:, node.feature] > node.threshold
        )

        is_left_leaf = (
            np.sum(left_population) < self.min_pop or
            node.depth + 1 == self.max_depth or
            len(np.unique(self.target[left_population])) == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (
            np.sum(right_population) < self.min_pop or
            node.depth + 1 == self.max_depth or
            len(np.unique(self.target[right_population])) == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Yeni bir yarpaq obyekti yaradır və müvafiq sinfi təyin edir"""
        if np.sum(sub_population) == 0:
            value = 0
        else:
            value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Yeni bir daxili düyün obyekti yaradır"""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Modelin verilən test dəsti üzərindəki dəqiqliyini hesablayır"""
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target
        )) / test_target.size

    def __str__(self):
        """Ağacı bütövlükdə sətir kimi vizuallaşdırır"""
        return self.root.__str__()
