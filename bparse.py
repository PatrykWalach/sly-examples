from typing import Any
from sly import Lexer
from sly import Parser


class BasicLexer(Lexer):
    tokens = {NAME, NUMBER, STRING, IF, THEN, ELSE, FOR, FUN, TO, ARROW, EQEQ}
    ignore = '\t '

    literals = {'=', '+', '-', '/', '*', '(', ')', ',', ';'}

    # Define tokens
    IF = r'IF'
    THEN = r'THEN'
    ELSE = r'ELSE'
    FOR = r'FOR'
    FUN = r'FUN'
    TO = r'TO'
    ARROW = r'->'
    NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'
    STRING = r'\".*?\"'

    EQEQ = r'=='

    @_(r'\d+')
    def NUMBER(self, t):
        t.value = int(t.value)
        return t

    @_(r'#.*')
    def COMMENT(self, t):
        pass

    @_(r'\n+')
    def newline(self, t):
        self.lineno = t.value.count('\n')


class BasicParser(Parser):
    tokens = BasicLexer.tokens

    precedence = (
        ('left', '+', '-'),
        ('left', '*', '/'),
        ('right', 'UMINUS'),
    )

    def __init__(self):
        self.env = {}

    @_('')
    def statement(self, p):
        pass

    @_('FOR var_assign TO expr THEN statement')
    def statement(self, p):
        return ('for_loop', ('for_loop_setup', p.var_assign, p.expr), p.statement)

    @_('IF condition THEN statement ELSE statement')
    def statement(self, p):
        return ('if_stmt', p.condition, ('branch', p.statement0, p.statement1))

    @_('FUN NAME "(" ")" ARROW statement')
    def statement(self, p):
        return ('fun_def', p.NAME, p.statement)

    @_('NAME "(" ")"')
    def statement(self, p):
        return ('fun_call', p.NAME)

    @_('expr EQEQ expr')
    def condition(self, p):
        return ('condition_eqeq', p.expr0, p.expr1)

    @_('var_assign')
    def statement(self, p):
        return p.var_assign

    @_('NAME "=" expr')
    def var_assign(self, p):
        return ('var_assign', p.NAME, p.expr)

    @_('NAME "=" STRING')
    def var_assign(self, p):
        return ('var_assign', p.NAME, p.STRING)

    @_('expr')
    def statement(self, p):
        return (p.expr)

    @_('expr "+" expr')
    def expr(self, p):
        return ('add', p.expr0, p.expr1)

    @_('expr "-" expr')
    def expr(self, p):
        return ('sub', p.expr0, p.expr1)

    @_('expr "*" expr')
    def expr(self, p):
        return ('mul', p.expr0, p.expr1)

    @_('expr "/" expr')
    def expr(self, p):
        return ('div', p.expr0, p.expr1)

    @_('"-" expr %prec UMINUS')
    def expr(self, p):
        return p.expr

    @_('NAME')
    def expr(self, p):
        return ('var', p.NAME)

    @_('NUMBER')
    def expr(self, p):
        return ('num', p.NUMBER)


class Visitor:
    env = {}

    def visit(self, node: Any) -> Any:
        match node:
            case ('num' | 'str', value):
                return self.visit(value)
            case ('if_stmt', left, (then, _else)):
                if self.visit(left):
                    return self.visit(then)
                return self.visit(_else)
            case ('fun_def', left, right):
                self.env[left] = right
            case ('fun_call', value):
                try:
                    return self.visit(self.env[value])
                except LookupError:
                    print("Undefined function '%s'" % node[1])
                return 0
            case ('add', left, right):
                return self.visit(left) + self.visit(right)
            case ('sub', left, right):
                return self.visit(left) - self.visit(right)
            case ('mul', left, right):
                return self.visit(left) * self.visit(right)
            case ('div', left, right):
                return self.visit(left) / self.visit(right)
            case ('var_assign', left, right):
                self.env[left] = self.visit(right)
                return left
            case ('var', value):
                try:
                    return self.env[value]
                except LookupError:
                    print("Undefined variable '%s'" % node[1])
                return 0
            case ('for_loop', ('for_loop_setup', left, right), body):
                left = self.visit(left)
                count = self.env[left]
                limit = self.visit(right)

                for i in range(count+1, limit+1):
                    self.visit(body)
                    self.env[left] = i

                del self.env[left]

            case _:
                return node


if __name__ == '__main__':
    lexer = BasicLexer()
    parser = BasicParser()
    while True:
        try:
            text = input('basic > ')
        except EOFError:
            break
        if text:
            tree = parser.parse(lexer.tokenize(text))
            print(tree)
            # match Visitor().visit(tree):
            #     case None: pass
            #     case some: print(some)
