/* expr.y - gramatică LR(1) pentru expresii aritmetice
   Folosită pentru a genera tabelele LR(1) cu Bison.
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

/* vrem fișier .output cu automatonul LR(1) */
%verbose

/* nu folosim valori semantice, dar punem ceva generic */
%define api.value.type {int}

/* token pentru identificatori / numere (id) */
%token ID

/* simbol de start */
%start E

%%

/* 1,2: reguli pentru E */
E
    : E '+' T      /* 1: E -> E + T */
    | T            /* 2: E -> T */
    ;

/* 3,4: reguli pentru T */
T
    : T '*' F      /* 3: T -> T * F */
    | F            /* 4: T -> F */
    ;

/* 5,6: reguli pentru F */
F
    : '(' E ')'    /* 5: F -> ( E ) */
    | ID           /* 6: F -> id */
    ;

%%

/* Lexer minimalist:
   - sare peste spatii
   - recunoaște + * ( )
   - orice alt caracter nenul îl tratează ca ID

   Pentru generarea tabelelor LR(1) e suficient.
*/
int yylex(void)
{
    int c;

    /* sarim peste spații/alb */
    do {
        c = getchar();
    } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');

    if (c == EOF)
        return 0;   /* 0 = EOF pentru Bison */

    if (c == '+')
        return '+';
    if (c == '*')
        return '*';
    if (c == '(')
        return '(';
    if (c == ')')
        return ')';

    /* orice altceva considerăm "id" */
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
