#ifndef __EXPRESSION_H__
#define __EXPRESSION_H__

enum token {
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
    INTEGER,
    INCLUDE,
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

#endif // __EXPRESSION_H__
