%token PLUS
%token MINUS
%token TIMES
%token DIVIDE
%token NUMBER
%token LPAREN
%token RPAREN

%%

expr
    : expr PLUS term
    | expr MINUS term
    | term
    ;

term
    : term TIMES factor
    | term DIVIDE factor
    | factor
    ;

factor
    : NUMBER
    | LPAREN expr RPAREN
    ;
%%