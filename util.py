W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purple


def printAttr(attr):
    if type(attr) == int:
        return f"{O}{attr}{W}"
    return f"{B}'{attr}'{W}"


def formatToken(tok):
    return f"Token({G}type{W}={printAttr(tok.type)}, {G}value{W}={printAttr(tok.value)}, {G}lineno{W}={printAttr(tok.lineno)}, {G}index{W}={printAttr(tok.index)})"


def formatError(e):
    return f"{R}{e}{W}"
