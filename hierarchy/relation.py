import abc
from collections import defaultdict
from typing import List, Optional, Set, Tuple
import networkx as nx


class RelationSource(abc.ABC):
    @abc.abstractmethod
    def get_left_for(self, uid_right):
        pass

    @abc.abstractmethod
    def get_right_for(self, uid_left):
        pass


class StaticRelationSource(RelationSource):
    def __init__(self, data):
        self.right_for_left = defaultdict(list)
        self.left_for_right = defaultdict(list)
        for (left, right) in data:
            self.right_for_left[left].append(right)
            self.left_for_right[right].append(left)

    def get_left_for(self, uid_right):
        return self.left_for_right.get(uid_right, set())

    def get_right_for(self, uid_left):
        return self.right_for_left.get(uid_left, set())


class Relation:
    def __init__(self, uid: str, sources: List[RelationSource]):
        self.uid: str = uid
        self.sources: List[RelationSource] = sources
        self._rgraph: nx.DiGraph = nx.DiGraph()
        self._ugraph: nx.Graph = nx.Graph()
        self._pairs: Set[Tuple[str, str]] = set()

    def pairs(self) -> Set[Tuple[str, str]]:
        return self._pairs

    def add_pairs(self, pairs):
        for pair in pairs:
            self._rgraph.add_edge(pair[1], pair[0])
            self._ugraph.add_edge(pair[1], pair[0])
        return True

    def rgraph(self) -> nx.DiGraph:
        return self._rgraph

    def ugraph(self) -> nx.Graph:
        return self._ugraph


class ClassRelations(object):

    def __init__(self, cfg, is_a=False):
        """
        parents - Dictionary mapping class labels to lists of parent class labels in the hierarchy.
        children - Dictionary mapping class labels to lists of children class labels in the hierarchy.
        is_a: If true, the relation is exported as child-parent tuples, not as parent-child tuples.
        hytpe: hierarchy type (fine / simple / coarse)
        """
        object.__init__(self)
        nodes_file = f'hierarchy/relations/{cfg.DATA.name}_{cfg.LOSS.htype}.nodes.txt'
        with open(nodes_file) as f:
            split = [
                str(x).strip().split(" ", maxsplit=1) for x in f
            ]
            self.nodes = dict([(int(num), name) for (num, name) in split])

        relations_file = f'hierarchy/relations/{cfg.DATA.name}_{cfg.LOSS.htype}.child-parent.txt'
        with open(relations_file) as f:
            split = [
                str(x).strip().split(" ", maxsplit=1) for x in f
            ]
            if is_a:
                split = [(parent, child) for (child, parent) in split]
            self.relation = [
                (self.nodes[int(child)], self.nodes[int(parent)])
                for (parent, child) in split
            ]

    def get_relation_graph(self):
        relation_sources = [StaticRelationSource(self.relation)]
        relations = Relation(
            "chia::Hyponymy",
            sources=relation_sources
        )
        relations.add_pairs(self.relation)
        return relations
