%option reentrant
%option noyywrap

%{
    #include <stdio.h>

    union yylval {
        int val;
    } yylval;

    enum token {
        BLANK,
        VALUEIDENT,
        CHOICEIDENT,
        VAR,
        DOC,
        CMPOP,
        INVALIDTOKEN,
        CODE,
        COUNTERKIND,
        BOOL,
        COUNTERVISIBILITY,
        AND,
        TRIGGER,
        WHEN,
        ALIAS,
        COUNTER,
        DEFINE,
        ENUM,
        EQUAL,
        FORALL,
        IN,
        IS,
        NOT,
        REQUIRE,
        REQUIRES,
        VALUE,
        END,
        SYMMETRIC,
        ANTISYMMETRIC,
        ARROW,
        COLON,
        COMMA,
        LPAREN,
        RPAREN,
        BITOR,
        OR,
        SETDEFKEY,
        SET,
        SUBSETOF,
        SETIDENT,
        BASE,
        DISJOINT,
        QUOTIENT,
        OF,
        DIVIDE,
    };

    // Indicates whether a counter sums or adds.
    enum counter_kind {
        ADD,
        MUL,
    };

    // Indicates how a counter exposes how its maximum value.
    // The variants are ordered by increasing amount of information available.
    enum counter_visibility {
        // Only the minimal value is computed and stored.
        NOMAX,
        // Both the min and max are stored, but only the min is exposed.
        HIDDENMAX,
        // Both the min and the max value are exposed.
        FULL,
    };

    enum cmp_op {
        LT,
        GT,
        LEQ,
        GEQ,
        EQ,
        NEQ,
    };

    enum set_def_key {
        ITEMTYPE,
        IDTYPE,
        ITEMGETTER,
        IDGETTER,
        ITER,
        FROMSUPERSET,
        PREFIX,
        NEWOBJS,
        REVERSE,
        ADDTOSET,
    };
%}

num [0-9]
alpha [a-zA-Z_]
alpha_num {alpha}|{num}

whitespace [ \t\r\n]+
comment "//"[^/\n][^\n]*|"//"
blank {whitespace}|{comment}
c_comment_beg "/*"
c_comment_end "*/"

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

choice_ident [a-z][a-z_0-9]*
value_ident [A-Z][A-Z_0-9]*
set_ident [A-Z][A-Za-z_0-9]*
var \${alpha_num}+
code \"[^\n\"]*\"
doc "///"

/* https://github.com/a-haas/Compilation/blob/6fe46a65d64decdb648feba790bb7b1d2980ca0a/yacc/setparser.l */

/* 6 Patterns § http://westes.github.io/flex/manual/Patterns.html#Patterns */
%%

{alias} { return ALIAS; }
{counter} { return COUNTER; }
{define} { return DEFINE; }
{enum} { return ENUM; }
{forall} { return FORALL; }
{in} { return IN; }
{is} { return IS; }
{not} { return NOT; }
{product} {yylval.val = MUL;  return COUNTERKIND; }
{require} { return REQUIRE; }
{requires} { return REQUIRES; }
{sum} {yylval.val = ADD;  return COUNTERKIND; }
{value} { return VALUE; }
{when} { return WHEN; }
{trigger} { return TRIGGER; }
{half} {yylval.val = NOMAX;  return COUNTERVISIBILITY; }
{hidden} {yylval.val = HIDDENMAX;  return COUNTERVISIBILITY; }
{base} { return BASE; }

{item_type} {yylval.val = ITEMTYPE;  return SETDEFKEY; }
{new_objs} {yylval.val = NEWOBJS;  return SETDEFKEY; }
{id_type} {yylval.val = IDTYPE;  return SETDEFKEY; }
{item_getter} {yylval.val = ITEMGETTER;  return SETDEFKEY; }
{id_getter} {yylval.val = IDGETTER;  return SETDEFKEY; }
{iter} {yylval.val = ITER;  return SETDEFKEY; }
{prefix} {yylval.val = PREFIX;  return SETDEFKEY; }
{reverse} {yylval.val = REVERSE;  return SETDEFKEY; }
{add_to_set} {yylval.val = ADDTOSET;  return SETDEFKEY; }
{from_superset} {yylval.val = FROMSUPERSET;  return SETDEFKEY; }
{set} { return SET; }
{of} { return OF; }
{subsetof} { return SUBSETOF; }
{disjoint} { return DISJOINT; }
{quotient} { return QUOTIENT; }
{true} {yylval.val = 1;  return BOOL; }
{false} {yylval.val = 0;  return BOOL; }

{colon} { return COLON; }
{comma} { return COMMA; }
{lparen} { return LPAREN; }
{rparen} { return RPAREN; }
{bit_or} { return BITOR; }
{or} { return OR; }
{and} { return AND; }
{gt} {yylval.val = GT;  return CMPOP; }
{lt} {yylval.val = LT;  return CMPOP; }
{ge} {yylval.val = GEQ;  return CMPOP; }
{le} {yylval.val = LEQ;  return CMPOP; }
{equals} {yylval.val = EQ;  return CMPOP; }
{not_equals} {yylval.val = NEQ;  return CMPOP; }
{equal} { return EQUAL; }
{doc} {}
{end} { return END; }
{symmetric} { return SYMMETRIC; }
{antisymmetric} { return ANTISYMMETRIC; }
{arrow} { return ARROW; }
{divide} { return DIVIDE; }

{var} { return VAR; }
{code} { return CODE; }
{choice_ident} { return CHOICEIDENT; }
{value_ident} { return VALUEIDENT; }
{set_ident} { return SETIDENT; }
{blank} { return BLANK; }
<<EOF>> { return EOF; }
. { return INVALIDTOKEN; }
%%
