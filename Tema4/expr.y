/* expr.y - gramatica LR(1) pentru expresii aritmetice cu +, -, *, / si uminus
   Folosita pentru a genera tabelele LR(1) si fisierul expr.output (Tema 4).
*/

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

/* vrem fisier .output cu automatonul LR(1) */
%verbose

/* nu folosim valori semantice, dar punem ceva generic */
%define api.value.type {int}

/* token pentru identificatori / numere (id) */
%token ID

/* simbol de start */
%start E

%%

/* reguli pentru E */
E
    : E '+' T      /* 1: E -> E + T */
    | E '-' T      /* 2: E -> E - T */
    | T            /* 3: E -> T */
    ;

/* reguli pentru T */
T
    : T '*' F      /* 4: T -> T * F */
    | T '/' F      /* 5: T -> T / F */
    | F            /* 6: T -> F */
    ;

/* reguli pentru F */
F
    : '(' E ')'    /* 7: F -> ( E ) */
    | '-' '(' E ')'/* 8: F -> - ( E ) */
    | ID           /* 9: F -> id */
    ;

%%

/* Lexer minimalist:
   - sare peste spatii
   - recunoaste + - * / ( )
   - orice alt caracter nenul il trateaza ca ID
*/
int yylex(void)
{
    int c;

    /* sarim peste spatii/alb */
    do {
        c = getchar();
    } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');

    if (c == EOF)
        return 0;   /* 0 = EOF pentru Bison */

    if (c == '+')
        return '+';
    if (c == '-')
        return '-';
    if (c == '*')
        return '*';
    if (c == '/')
        return '/';
    if (c == '(')
        return '(';
    if (c == ')')
        return ')';

    /* orice altceva consideram "id" */
    return ID;
}

int main(void)
{
    printf("Introdu o expresie (ex: id+-(id*id)) si apasa Enter:\n");
    if (yyparse() == 0)
        printf("Expresie corecta.\n");
    else
        printf("Expresie incorecta.\n");
    return 0;
}
