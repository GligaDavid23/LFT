%{
    #include <stdio.h>
    #include <stdlib.h>

    int yylex(void);
    int yyerror(const char *s)
    {
        fprintf(stderr, "Eroare sintactica: %s\n", s);
        return 0;
    }
%}

%verbose

%define api.value.type {int}

%token ID

%start E

%%

E
    : E '+' T
    | E '-' T
    | T
    ;

T
    : T '*' F
    | F
    ;

F
    : '(' E ')'
    | ID
    ;

%%


int yylex(void)
{
    int c;

    do {
        c = getchar();
    } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');

    if (c == EOF)
        return 0;

    if (c == '+')
        return '+';
    if (c == '-')
        return '-';
    if (c == '*')
        return '*';
    if (c == '(')
        return '(';
    if (c == ')')
        return ')';

    return ID;
}

int main(void)
{
    printf("Introdu o expresie (ex: id+(id*id)) si apasa Enter:\n");
    if (yyparse() == 0)
        printf("Expresie corecta.\n");
    else
        printf("Expresie incorecta.\n");
    return 0;
}
