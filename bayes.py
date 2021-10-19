from __future__ import annotations
import functools
import json
import operator
import random
import dataclasses as dc
from functools import partial, reduce, wraps
from itertools import product
from typing import Any, Callable, Generic, TypeAlias, TypeVar, Union
from sly import Lexer, Parser


def _(fn: str, *args: list[str]) -> Callable[[R], R]:
    raise Exception


def parents(edges: list[tuple[str, A]], vertex: A):
    return [a for a, d in edges if d == vertex]


class BayesianNetwork:
    def __init__(self, *edges_or_vertices: str | tuple[str, str]):
        self.edges = [
            e for e in edges_or_vertices if isinstance(e, tuple)]

        self.vertices = set(sum([[*e] if isinstance(e, tuple) else [e]
                                 for e in edges_or_vertices], []))

        self.P = {
            e: {} for e in self.vertices
        }

    def parents(self, vertex: str):
        return parents(self.edges, vertex)

    def query(self, query: Query):
        p = self.predict(query_to_event_external_visitor(query))

        match query:
            case Query(Conditional(_, right)):
                return p/self.predict(query_to_event_external_visitor(Query(right)))
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


class AstEncoder(json.JSONEncoder):
    def default(self, obj):
        if dc.is_dataclass(obj):
            return {"type": type(obj).__name__} | {field: getattr(obj, field) for field in obj.__dataclass_fields__}
        return json.JSONEncoder.default(self, obj)


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
    def NUMBER(self, t):
        t.value = float(t.value)
        return t

    @_(r'\n+')
    def ignore_newline(self, t):
        self.lineno += len(t.value)


class BayesParser(Parser):
    tokens = BayesLexer.tokens
    debugfile = 'parser.out'

    def __init__(self) -> None:
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
        return Dependency(p.node0, p.node1)

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
        return Conditional(p.node, p.negated_nodes)

    @_('negated_node "|" negated_nodes')
    def conditional(self, p):
        return Conditional(p.negated_node, p.negated_nodes)

    @_('negated_node "," negated_nodes')
    def negated_nodes(self, p):
        return [p.negated_node, *p.negated_nodes]

    @_('negated_node')
    def negated_nodes(self, p):
        return [p.negated_node]

    @_('"~" negated_node')
    def negated_node(self, p):
        return Negation(p.negated_node)

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


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
E = TypeVar('E')
F = TypeVar('F')
G = TypeVar('G')
H = TypeVar('H')
I = TypeVar('I')
J = TypeVar('J')
K = TypeVar('K')
L = TypeVar('L')
M = TypeVar('M')
R = TypeVar('R')


@dc.dataclass(frozen=True)
class Node:
    value: str


T1 = TypeVar('T1')


@dc.dataclass(frozen=True)
class Negation(Generic[A]):
    value: A


@dc.dataclass(frozen=True)
class Dependency(Generic[B]):
    left: B
    right: B


@dc.dataclass(frozen=True)
class Conditional(Generic[D, E]):
    left: D
    right: list[E]


@dc.dataclass(frozen=True)
class Query(Generic[F, G]):
    value: F | list[G]


@dc.dataclass(frozen=True)
class Probability(Generic[H]):
    expression: H
    value: float


@dc.dataclass(frozen=True)
class Declaration(Generic[J, K, L]):
    nodes: list[J]
    conditional_dependencies: list[K]
    probabilities: list[L]


def internal(fn: Callable[[AstNode[R]], R]) -> Callable[[AstTree], R]:

    @wraps(fn)
    def visit(node: Any):
        if dc.is_dataclass(node):
            dataclass_values = (getattr(node, field)
                                for field in node.__dataclass_fields__)
            dataclass_values = [visit(value)
                                for value in dataclass_values]
            return fn(node.__class__(*dataclass_values))
        match node:
            case [*arr]:
                return [visit(v) for v in arr]
            case _:
                return node

    return visit


def external(fn: Callable[[AstNode[Callable[[], R]]], R]) -> Callable[[AstTree], R]:

    @wraps(fn)
    def visit(node: Any):
        if dc.is_dataclass(node):
            dataclass_values = (getattr(node, field)
                                for field in node.__dataclass_fields__)
            dataclass_values = [partial(visit, node=value)
                                for value in dataclass_values]
            return fn(node.__class__(*dataclass_values))
        match node:
            case [*arr]:
                return [partial(visit, node=v) for v in arr]
            case _:
                return node

    return visit


Event: TypeAlias = dict[str, bool]


AstNode: TypeAlias = Declaration[R, R, R] | Conditional[R,
                                                        R] | Node | Negation[R] | Query[R, R] | Dependency[R] | Probability[R]


AstNegation: TypeAlias = Negation[Union[Node, 'AstNegation']]
AstConditional: TypeAlias = Conditional[Node | AstNegation, Node | AstNegation]
AstQuery: TypeAlias = Query[AstConditional, Node | AstNegation]
AstDependency: TypeAlias = Dependency[Node]
AstProbability: TypeAlias = Probability[AstConditional]
AstDeclaration: TypeAlias = Declaration[Node, AstDependency, AstProbability]

AstTree: TypeAlias = AstNegation | AstConditional | AstDeclaration | AstQuery | AstDependency | Node | AstProbability


@internal
def query_to_event_visitor(query: AstNode[Event]) -> Event:
    match query:
        case Node(value):
            return {value: True}
        case Negation(node):
            return {key: not value for key, value in node.items()}
        case Conditional(left, list() as right):
            return functools.reduce(operator.or_, [left, *right])
        case Query([*arr]):
            return functools.reduce(operator.or_, arr)
        case Query(value) if not isinstance(value, list):
            return value
        case err:
            raise TypeError(err)


@external
def query_to_event_external_visitor(query: AstNode[Callable[[], Event]]) -> Event:
    match query:
        case Node(value):
            return {value: True}
        case Negation(node):
            return {key: not value for key, value in node().items()}
        case Conditional(left, list() as right):
            left_and_right = [fn() for fn in (left, *right)]
            return functools.reduce(operator.or_, left_and_right)
        case Query([*arr]):
            arr = [fn() for fn in arr]
            return functools.reduce(operator.or_, arr)
        case Query(value) if not isinstance(value, list):
            return value()
        case node:
            raise TypeError(node)


@internal
def remove_negation_visitor(node):
    match node:
        case Negation(Negation(node)):
            return node
        case _:
            return node


@internal
def get_node_visitor(any_node: AstNode[Node]) -> Node:
    match any_node:
        case Node():
            return any_node
        case Negation(node):
            return node
        case _:
            raise TypeError(any_node)


def main():
    with open('bayes.txt') as f:
        text = f.read()

    lexer = BayesLexer()
    parser = BayesParser()

    any_declaration = parser.parse(lexer.tokenize(text))

    match any_declaration:
        case Declaration([*_], [*_], [*_]) as d:
            declaration: AstDeclaration = d
        case _:
            raise ValueError

    declaration = remove_negation_visitor(declaration)

    with open('bayes-ast.json', 'w') as f:
        json.dump(declaration, f, cls=AstEncoder, indent=4)

    edges = [node.value for node in declaration.nodes]

    verticies = [(a.left.value, a.right.value)
                 for a in declaration.conditional_dependencies]

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

    match probability.expression:
        case Conditional(Negation(value) | Node(value)):
            vertex = value
        case _:
            raise ValueError(f'Invalid conditional probability: {probability}')

    keys = query_to_event_visitor(
        Query(probability.expression)).items()

    n.P[vertex][tuple(k for _, k in sorted(keys))] = probability.value
    n.P[vertex][tuple(k if v != vertex else not k for v,
                      k in sorted(keys))] = 1-probability.value


if __name__ == '__main__':
    main()
