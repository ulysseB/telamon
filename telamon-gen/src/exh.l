/* 19 Reentrant C Scanners § https://westes.github.io/flex/manual/Reentrant.html#Reentrant */
%option reentrant
/* https://westes.github.io/flex/manual/I-get-an-error-about-undefined-yywrap_0028_0029_002e.html#I-get-an-error-about-undefined-yywrap_0028_0029_002e */
%option noyywrap
/* number of the current line § https://westes.github.io/flex/manual/Options-Affecting-Scanner-Behavior.html#index-yylineno */
%option yylineno
/* 19.4.6 Extra Data § https://westes.github.io/flex/manual/Extra-Data.html */
%option extra-type="Span"

%x C_COMMENT
%x LINE_DOC
%x LINE_CODE

%{
    #include "expression.h"

    typedef struct Pos {
        unsigned int line;
        unsigned int column;
    } Pos;

    typedef int Data;

    // 
    typedef struct Span {
        Pos beg;
        Pos end;
        Data data;
    } Span;

    #define YY_USER_ACTION { \
        yyextra.beg = yyextra.end; \
        yyextra.end.line = yylineno; \
        yyextra.end.column += yyleng; \
    }

%}

num [0-9]
alpha [a-zA-Z_]
alpha_num {alpha}|{num}

whitespace [ \t\r]+
comment "//"[^/\n][^\n]*|"//"
blank {comment}|{whitespace}

c_comment_beg "/*"
c_comment_end "*/"

doc "///"

alias "alias"
counter "counter"
define "define"
enum "enum"
forall "forall"
in "in"
is "is"
not "not"
product "mul"
require "require"
requires "requires"
sum "sum"
value "value"
end "end"
symmetric "symmetric"
antisymmetric "antisymmetric"
arrow "->"
when "when"
trigger "trigger"
half "half"
hidden "internal"
base "base"

set "set"
subsetof "subsetof"
item_type "item_type"
id_type "id_type"
item_getter "item_getter"
id_getter "id_getter"
iter "iterator"
from_superset "from_superset"
add_to_set "add_to_set"
prefix "var_prefix"
new_objs "new_objs"
disjoint "disjoint"
reverse "reverse"
quotient "quotient"
of "of"
true "true"
false "false"

colon ":"
comma ","
lparen "("
rparen ")"
bit_or "|"
or "||"
and "&&"
gt ">"
lt "<"
ge ">="
le "<="
equals "=="
not_equals "!="
equal "="
divide "/"
integer "integer"
include "include"{whitespace}*\".*\"

choice_ident [a-z][a-z_0-9]*
value_ident [A-Z][A-Z_0-9]*
set_ident [A-Z][A-Za-z_0-9]*
var \${alpha_num}+

/* Patterns documentation: http://westes.github.io/flex/manual/Patterns.html#Patterns */
%%

{c_comment_beg}            { BEGIN(C_COMMENT); }
<C_COMMENT>[\n]            { yyextra.end.line += 1; }
<C_COMMENT>{c_comment_end} { BEGIN(INITIAL); }
<C_COMMENT>.               { }

{doc}                      { yyextra.end.column -= 3; BEGIN(LINE_DOC); }
<LINE_DOC>\n               { yyextra.end.column = 0; BEGIN(INITIAL); }
<LINE_DOC>[^\n]*           { yyextra.end.column += 3; return DOC; }

\"                         { BEGIN(LINE_CODE); }
<LINE_CODE>\"              { BEGIN(INITIAL); }
<LINE_CODE>\\\n[ \t]*      { yyextra.end.column -= strlen(yytext)+1; }
<LINE_CODE>[^\\"]+|\\      { return CODE; }

[\n] { yyextra.end.column = 0; }

{blank} { }

{include} {
    int length = strlen(yytext);
    yytext[length-1] = '\0';
    yytext = memchr(yytext, '\"', length)+1;
    return INCLUDE;
}

{alias} { return ALIAS; }
{counter} { return COUNTER; }
{define} { return DEFINE; }
{enum} { return ENUM; }
{forall} { return FORALL; }
{in} { return IN; }
{is} { return IS; }
{not} { return NOT; }
{product} { yyextra.data = MUL;  return COUNTERKIND; }
{require} { return REQUIRE; }
{requires} { return REQUIRES; }
{sum} { yyextra.data = ADD;  return COUNTERKIND; }
{value} { return VALUE; }
{when} { return WHEN; }
{trigger} { return TRIGGER; }
{half} { yyextra.data = NOMAX;  return COUNTERVISIBILITY; }
{hidden} { yyextra.data = HIDDENMAX;  return COUNTERVISIBILITY; }
{base} { return BASE; }

{item_type} { yyextra.data = ITEMTYPE;  return SETDEFKEY; }
{new_objs} { yyextra.data = NEWOBJS;  return SETDEFKEY; }
{id_type} { yyextra.data = IDTYPE;  return SETDEFKEY; }
{item_getter} { yyextra.data = ITEMGETTER;  return SETDEFKEY; }
{id_getter} { yyextra.data = IDGETTER;  return SETDEFKEY; }
{iter} { yyextra.data = ITER;  return SETDEFKEY; }
{prefix} { yyextra.data = PREFIX;  return SETDEFKEY; }
{reverse} { yyextra.data = REVERSE;  return SETDEFKEY; }
{add_to_set} { yyextra.data = ADDTOSET;  return SETDEFKEY; }
{from_superset} { yyextra.data = FROMSUPERSET;  return SETDEFKEY; }
{set} { return SET; }
{of} { return OF; }
{subsetof} { return SUBSETOF; }
{disjoint} { return DISJOINT; }
{quotient} { return QUOTIENT; }
{true} { yyextra.data = 1; return BOOL; }
{false} { yyextra.data = 0; return BOOL; }

{colon} { return COLON; }
{comma} { return COMMA; }
{lparen} { return LPAREN; }
{rparen} { return RPAREN; }
{bit_or} { return BITOR; }
{or} { return OR; }
{and} { return AND; }
{gt} { yyextra.data = GT; return CMPOP; }
{lt} { yyextra.data = LT; return CMPOP; }
{ge} { yyextra.data = GEQ; return CMPOP; }
{le} { yyextra.data = LEQ; return CMPOP; }
{equals} { yyextra.data = EQ; return CMPOP; }
{not_equals} { yyextra.data = NEQ; return CMPOP; }
{equal} { return EQUAL; }
{end} { return END; }
{symmetric} { return SYMMETRIC; }
{antisymmetric} { return ANTISYMMETRIC; }
{arrow} { return ARROW; }
{divide} { return DIVIDE; }
{integer} { return INTEGER; }

{var} { yytext++; return VAR; }
{choice_ident} { return CHOICEIDENT; }
{value_ident} { return VALUEIDENT; }
{set_ident} { return SETIDENT; }
<<EOF>> { return EOF; }
. { return INVALIDTOKEN; }
%%
