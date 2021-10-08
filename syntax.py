from sly import Lexer, Parser


class BaseLexer(Lexer):
    tokens = {NUMBER, ADDITIVE_OPERATOR,
              MULTIPLICATIVE_OPERATOR}
    ignore = r'\s+'

    literals = { '(', ')' }


    NUMBER = r'\d+'

    ADDITIVE_OPERATOR = r'[+\-]'
    MULTIPLICATIVE_OPERATOR = r'[*/]'


class BaseParser(Parser):
    tokens = BaseLexer.tokens

    @_('additive_expression')
    def expression(self, p):
        return p.additive_expression

    @_('additive_expression ADDITIVE_OPERATOR multiplicative_expression')
    def additive_expression(self, p):
        return {
            'type': 'BinaryExpression',
            "op": p.ADDITIVE_OPERATOR,
            "left": p.additive_expression,
            "right": p.multiplicative_expression
        }

    @_('multiplicative_expression')
    def additive_expression(self, p):
        return p.multiplicative_expression

    @_('multiplicative_expression MULTIPLICATIVE_OPERATOR primary_expression')
    def multiplicative_expression(self, p):
        return {
            'type': 'BinaryExpression',
            "op": p.MULTIPLICATIVE_OPERATOR,
            "left": p.multiplicative_expression,
            "right": p.primary_expression
        }

    @_('primary_expression')
    def multiplicative_expression(self, p):
        return p.primary_expression

    @_('literal')
    def primary_expression(self, p):
        return p.literal

    @_('parenthesized_expression')
    def primary_expression(self, p):
        return p.parenthesized_expression

    @_('numeric_literal')
    def literal(self, p):
        return p.numeric_literal

    @_('NUMBER')
    def numeric_literal(self, p):
        return {'type': 'Literal', 'raw': p.NUMBER, 'value': float(p.NUMBER)}

    @_('"(" additive_expression ")"')
    def parenthesized_expression(self, p):
        return p.additive_expression


if __name__ == '__main__':
    lexer = BaseLexer()
    parser = BaseParser()

    text = '2+2'

    parser.parse(lexer.tokenize(text))
