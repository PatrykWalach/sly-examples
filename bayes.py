from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools as ft
import functools
import itertools
from sly import Lexer, Parser
import random
import re
from itertools import product
from functools import partial, reduce
import operator


def parents(edges: list[tuple[str, str]], vertex: str):
    return [a for a, d in edges if d == vertex]


class BayesianNetwork:
    def __init__(self, *edges_or_vertices):
        self.edges = [e for e in edges_or_vertices if isinstance(e, tuple)]

        self.vertices = set(sum([[*e] if isinstance(e, tuple) else [e]
                                 for e in edges_or_vertices], []))

        self.P = {
            e: {} for e in self.vertices
        }

    def parents(self, vertex: str):
        return parents(self.edges, vertex)

    def query(self, query: Query):
        p = self.predict(query_to_event(query))

        match query:
            case Query(Conditional(_, right)):
                return p/self.predict(query_to_event(Query(right)))
            case Query([*_]):
                return p

    def verify(self):
        for v in self.vertices:
            for k in product([True, False], repeat=len(self.parents(v))):
                if (*k, True) not in self.P[v]:
                    parents = [e if s else "~"+e for e,
                               s in zip(self.parents(v), k)]
                    raise ValueError(
                        f'Not enough probabilities for event {v}\nPlease add "P({v}{"|"+",".join(parents) if len(parents) else ""})=n"')

    def probability(self, of: tuple, event: dict):
        k, value = of

        if k not in self.P:
            raise ValueError(f'"{k}" not in the network')

        return self.P[k][tuple(v for k, v in sorted([*filter(lambda i: i[0] in self.parents(k), event.items()), of]))]

    def predict(self,  event: dict):
        """return P(A,~B) as float
        from event = { 'A': True, 'B': False }
        """

        missing_vertices = set(self.vertices).difference(event.keys())

        if len(missing_vertices):
            return sum(
                self.predict({
                    **event,
                    **dict(zip(missing_vertices, values)),
                })
                for values in product([False, True],  repeat=len(missing_vertices))
            )

        apply_event = partial(self.probability, event=event)

        probabilities = map(apply_event, event.items())

        return reduce(operator.mul, probabilities, 1)


class BayesContext:
    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.verticies: list[tuple[str, str]] = []


class BayesLexer(Lexer):
    tokens = {NODE, P, NODES, NUMBER,
              CONDITIONAL_DEPENDENCIES,  PROBABILITIES, }

    ignore = ' \t'

    literals = {'(', ')', '=', '|', '~', ','}

    NODES = 'Nodes:'
    CONDITIONAL_DEPENDENCIES = 'Conditional Dependencies:'
    PROBABILITIES = 'Probabilities:'

    NODE = r'[A-Z]+'

    NODE['P'] = P

    @_(r'\d+\.\d+')
    def NUMBER(self, p):
        p.value = float(p.value)
        return p

    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)


class BayesParser(Parser):
    tokens = BayesLexer.tokens
    debugfile = 'parser.out'

    def __init__(self) -> None:
        self.ctx = BayesContext()
        super().__init__()

    @_('query')
    def expr(self, p):
        return p.query

    @_('declaration')
    def expr(self, p):
        return p.declaration

    @_('NODES nodes_declaration optional_conditional_dependencies PROBABILITIES probabilities')
    def declaration(self, p):
        return Declaration(p.nodes_declaration, p.optional_conditional_dependencies, p.probabilities)

    @_('"P" "(" conditional ")" "="')
    def query(self, p):
        return Query(p.conditional)

    @_('"P" "(" negated_nodes ")" "="')
    def query(self, p):
        return Query(p.negated_nodes)

    @_('nodes')
    def nodes_declaration(self, p):
        self.ctx.nodes = [node.value for node in p.nodes]
        return p.nodes

    @_('node "," nodes')
    def nodes(self, p):
        return [p.node, *p.nodes]

    @_('node')
    def nodes(self, p):
        return [p.node]

    @_('CONDITIONAL_DEPENDENCIES conditional_dependencies')
    def optional_conditional_dependencies(self, p):
        return p.conditional_dependencies

    @_('')
    def optional_conditional_dependencies(self, p):
        return []

    @_('conditional_dependency "," conditional_dependencies')
    def conditional_dependencies(self, p):
        return [p.conditional_dependency, *p.conditional_dependencies]

    @_('conditional_dependency')
    def conditional_dependencies(self, p):
        return [p.conditional_dependency]

    @_('"(" node "," node ")"')
    def conditional_dependency(self, p):
        self.ctx.verticies.append((p.node0.value, p.node1.value))
        return (p.node0, p.node1)

    @_('probability probabilities')
    def probabilities(self, p):
        return [p.probability, *p.probabilities]

    @_('probability')
    def probabilities(self, p):
        return [p.probability]

    @_('"P" "(" probability_conditional ")" "=" NUMBER')
    def probability(self, p):
        return Probability(p.probability_conditional, p.NUMBER)

    @_('node')
    def probability_conditional(self, p):
        return Conditional(p.node, [])

    @_('node "|" negated_nodes')
    def probability_conditional(self, p):
        node_parents = parents(self.ctx.verticies, p.node.value)
        negated_nodes = {node.value: node for node in p.negated_nodes}

        for (node_name, node) in negated_nodes.items():
            if node_name not in node_parents:
                print(
                    f'warning: "{p.node}" does not depend on "{node_name}"')
                p.negated_nodes.remove(node)

        for missing_parent in set(node_parents).difference(set(negated_nodes.keys())):
            raise ValueError(
                f'Incorrect probability: {p.node}" also depends on "{missing_parent}"'
            )

        return Conditional(p.node, p.negated_nodes)

    @_('negated_node "|" negated_nodes')
    def conditional(self, p):
        return Conditional(p.negated_node, p.negated_nodes)

    @_('negated_node "," negated_nodes')
    def negated_nodes(self, p):
        if p.negated_node in p.negated_nodes:
            print(f'warning: Multiple nodes: "{p.negated_node}"')
            return p.negated_nodes

        return [p.negated_node, *p.negated_nodes]

    @_('negated_node')
    def negated_nodes(self, p):
        return [p.negated_node]

    @_('"~" negated_node')
    def negated_node(self, p):
        match p.negated_node:
            case NegatedNode(value):
                return Node(value)
            case Node(value):
                return NegatedNode(value)

    @_('node')
    def negated_node(self, p):
        return p.node

    @_('NODE')
    def node(self, p):
        return Node(p.NODE)


def parse_query(logic: str):
    """return AST ('|', 'H', (',', ('~', 'MA'), ('~', 'GR')))
    from query = 'P(H|~MA,GR)='
    """
    parser = BayesParser()
    lexer = BayesLexer()
    return parser.parse(lexer.tokenize(logic))


class AstNode(ABC):
    @abstractmethod
    def accept(self, visitor: Visitor):
        return NotImplemented


@dataclass
class Node(AstNode):
    value: str

    def accept(self, visitor: Visitor):
        visitor.visitNode(self)


@dataclass
class NegatedNode(AstNode):
    value: str

    def accept(self, visitor: Visitor):
        visitor.visitNegatedNode(self)


@dataclass
class Conditional(AstNode):
    negated_node: NegatedNode | Node
    negated_nodes: list[NegatedNode | Node]

    def accept(self, visitor: Visitor):
        for v in self.negated_nodes:
            v.accept(visitor)

            self.negated_node.accept(visitor)

        visitor.visitConditional(self)


@dataclass
class Query(AstNode):
    value: Conditional | list[NegatedNode | Node]

    def accept(self, visitor: Visitor):
        match self.value:
            case [*arr]:
                for v in arr:
                    v.accept(visitor)
            case v:
                v.accept(visitor)

        visitor.visitQuery(self)


@dataclass
class Probability(AstNode):
    of: Conditional
    value: float

    def accept(self, visitor: Visitor):
        self.of.accept(visitor)
        visitor.visitProbability(self)


@dataclass
class Declaration(AstNode):
    nodes: list[Node]
    conditional_dependencies: list[tuple[Node, Node]]
    probabilities: list[Probability]

    def accept(self, visitor: Visitor):
        for v in self.nodes:
            v.accept(visitor)
        for v in self.conditional_dependencies:
            v.accept(visitor)
        for v in self.probabilities:
            v.accept(visitor)

        visitor.visitDeclaration(self)


class Visitor(ABC):
    def visitNode(self, node: Node) -> None:
        pass

    def visitNegatedNode(self, node: NegatedNode) -> None:
        pass

    def visitConditional(self, node: Conditional) -> None:
        pass

    def visitQuery(self, node: Query) -> None:
        pass

    def visitProbability(self, node: Probability) -> None:
        pass

    def visitDeclaration(self, node: Declaration) -> None:
        pass


class QueryToEventVisitor(Visitor):
    def __init__(self) -> None:
        self.event: dict[str, bool] = {}
        super().__init__()

    def visitNode(self, node: Node):
        self.event[node.value] = True

    def visitNegatedNode(self, node: NegatedNode):
        self.event[node.value] = False


def query_to_event(query: Query) -> dict[str, bool]:
    visitor = QueryToEventVisitor()
    query.accept(visitor)
    return visitor.event


def main():
    with open('bayes.txt') as f:
        text = f.read()

    lexer = BayesLexer()
    parser = BayesParser()

    declaration: Declaration = parser.parse(lexer.tokenize(text))

    edges = [node.value for node in declaration.nodes]

    verticies = [(a.value, b.value)
                 for a, b in declaration.conditional_dependencies]

    n = BayesianNetwork(*edges, *verticies)

    for initial_probability in declaration.probabilities:
        apply_probability(n, initial_probability)

    n.verify()

    v = random.choice([*n.vertices])

    i = input(
        f'Please input queries, for example: "P({v}{"|"+",".join(n.parents(v)) if len(n.parents(v)) else ""})="\n')
    while(True):
        m = parser.parse(lexer.tokenize(i))
        try:
            print(n.query(m))
        except ValueError as e:
            print(e)
        i = input()


def apply_probability(n: BayesianNetwork, probability: Probability):

    match probability.of:
        case Conditional(NegatedNode(value) | Node(value)):
            vertex = value
        case _:
            raise ValueError(f'Invalid conditional probability: {probability}')

    keys = query_to_event(Query(probability.of)).items()

    n.P[vertex][tuple(k for _, k in sorted(keys))] = probability.value
    n.P[vertex][tuple(k if v != vertex else not k for v,
                      k in sorted(keys))] = 1-probability.value


if __name__ == '__main__':
    main()
