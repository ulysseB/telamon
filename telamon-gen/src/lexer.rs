/// Tokens from the textual representation of constraints.
use ir;
use std;

#[derive(Debug)]
pub enum Token {
    ValueIdent(String), ChoiceIdent(String), Var(String), Doc(String), CmpOp(ir::CmpOp),
    InvalidToken(String), Code(String), CounterKind(ir::CounterKind), Bool(bool),
    CounterVisibility(ir::CounterVisibility),
    And, Trigger, When, Alias, Counter, Define, Enum, Equal, Forall, In, Is, Not, Require,
    Requires, Value, End, Symmetric, AntiSymmetric, Arrow, Colon, Comma, LParen, RParen,
    BitOr, Or, SetDefKey(ir::SetDefKey), Set, SubsetOf, SetIdent(String), Base, Disjoint,
    Quotient, Of, Divide,
}

rustlex! Lexer {
    property comment: String = String::new();

    let NUM = ['0'-'9'];
    let ALPHA = ['a'-'z''A'-'Z''_'];
    let ALPHA_NUM = ALPHA | NUM;

    let WHITESPACE = [' ''\t''\r''\n']+;
    let COMMENT = "//"[^'/''\n'][^'\n']* | "//";
    let BLANK = WHITESPACE | COMMENT;
    let C_COMMENT_BEG = "/*";
    let C_COMMENT_END = "*/";

    let ALIAS = "alias";
    let COUNTER = "counter";
    let DEFINE = "define";
    let ENUM = "enum";
    let FORALL = "forall";
    let IN = "in";
    let IS = "is";
    let NOT = "not";
    let PRODUCT = "mul";
    let REQUIRE = "require";
    let REQUIRES = "requires";
    let SUM = "sum";
    let VALUE = "value";
    let END = "end";
    let SYMMETRIC = "symmetric";
    let ANTISYMMETRIC = "antisymmetric";
    let ARROW = "->";
    let WHEN = "when";
    let TRIGGER = "trigger";
    let HALF = "half";
    let HIDDEN = "internal";
    let BASE = "base";

    let SET = "set";
    let SUBSETOF = "subsetof";
    let ITEM_TYPE = "item_type";
    let ID_TYPE = "id_type";
    let ITEM_GETTER = "item_getter";
    let ID_GETTER = "id_getter";
    let ITER = "iterator";
    let FROM_SUPERSET = "from_superset";
    let ADD_TO_SET = "add_to_set";
    let PREFIX = "var_prefix";
    let NEW_OBJS = "new_objs";
    let DISJOINT = "disjoint";
    let REVERSE = "reverse";
    let QUOTIENT = "quotient";
    let OF = "of";
    let TRUE = "true";
    let FALSE = "false";

    let COLON = ":";
    let COMMA = ",";
    let LPAREN = "(";
    let RPAREN = ")";
    let BIT_OR = "|";
    let OR = "||";
    let AND = "&&";
    let GT = ">";
    let LT = "<";
    let GE = ">=";
    let LE = "<=";
    let EQUALS = "==";
    let NOT_EQUALS = "!=";
    let EQUAL = "=";
    let DIVIDE = "/";

    let CHOICE_IDENT = ['a'-'z']['a'-'z''_''0'-'9']*;
    let VALUE_IDENT = ['A'-'Z']['A'-'Z''_''0'-'9']*;
    let SET_IDENT = ['A'-'Z']['A'-'Z''a'-'z''_''0'-'9']*;
    let VAR = '$' ALPHA_NUM+;
    let CODE = '\"'[^'\n''\"']*'\"';
    let DOC = "///";

    INITIAL {
        . => |lex: &mut Lexer<R>| Some(Token::InvalidToken(lex.yystr())),
        BLANK => |_: &mut Lexer<R>| None,
        C_COMMENT_BEG => |lex: &mut Lexer<R>| { lex.COMMENT_MODE(); None },
        CHOICE_IDENT => |lex: &mut Lexer<R>| Some(Token::ChoiceIdent(lex.yystr())),
        SET_IDENT => |lex: &mut Lexer<R>| Some(Token::SetIdent(lex.yystr())),
        VALUE_IDENT => |lex: &mut Lexer<R>| Some(Token::ValueIdent(lex.yystr())),
        VAR => |lex: &mut Lexer<R>| Some(Token::Var(lex.yystr()[1..].to_string())),
        CODE => |lex: &mut Lexer<R> | {
            let mut code = lex.yystr()[1..].to_string();
            code.pop();
            Some(Token::Code(code))
        },
        ALIAS => |_: &mut Lexer<R>| Some(Token::Alias),
        COUNTER => |_: &mut Lexer<R>| Some(Token::Counter),
        DEFINE => |_: &mut Lexer<R>| Some(Token::Define),
        ENUM => |_: &mut Lexer<R>| Some(Token::Enum),
        FORALL => |_: &mut Lexer<R>| Some(Token::Forall),
        IN => |_: &mut Lexer<R>| Some(Token::In),
        IS => |_: &mut Lexer<R>| Some(Token::Is),
        NOT => |_: &mut Lexer<R>| Some(Token::Not),
        PRODUCT => |_: &mut Lexer<R>| Some(Token::CounterKind(ir::CounterKind::Mul)),
        REQUIRE => |_: &mut Lexer<R>| Some(Token::Require),
        REQUIRES => |_: &mut Lexer<R>| Some(Token::Requires),
        SUM => |_: &mut Lexer<R>| Some(Token::CounterKind(ir::CounterKind::Add)),
        VALUE => |_: &mut Lexer<R>| Some(Token::Value),
        WHEN => |_: &mut Lexer<R>| Some(Token::When),
        TRIGGER => |_: &mut Lexer<R>| Some(Token::Trigger),
        HALF => |_: &mut Lexer<R>|
            Some(Token::CounterVisibility(ir::CounterVisibility::NoMax)),
        HIDDEN => |_: &mut Lexer<R>|
            Some(Token::CounterVisibility(ir::CounterVisibility::HiddenMax)),
        BASE => |_: &mut Lexer<R>| Some(Token::Base),

        ITEM_TYPE =>  |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::ItemType)),
        NEW_OBJS =>  |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::NewObjs)),
        ID_TYPE => |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::IdType)),
        ITEM_GETTER => |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::ItemGetter)),
        ID_GETTER => |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::IdGetter)),
        ITER => |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::Iter)),
        PREFIX => |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::Prefix)),
        REVERSE => |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::Reverse)),
        ADD_TO_SET => |_: &mut Lexer<R>| Some(Token::SetDefKey(ir::SetDefKey::AddToSet)),
        FROM_SUPERSET => |_: &mut Lexer<R>|
            Some(Token::SetDefKey(ir::SetDefKey::FromSuperset)),
        SET => |_: &mut Lexer<R>| Some(Token::Set),
        SUBSETOF => |_: &mut Lexer<R>| Some(Token::SubsetOf),
        DISJOINT => |_: &mut Lexer<R>| Some(Token::Disjoint),
        QUOTIENT => |_: &mut Lexer<R>| Some(Token::Quotient),
        OF => |_: &mut Lexer<R>| Some(Token::Of),
        TRUE => |_: &mut Lexer<R>| Some(Token::Bool(true)),
        FALSE => |_: &mut Lexer<R>| Some(Token::Bool(false)),


        COLON => |_: &mut Lexer<R>| Some(Token::Colon),
        COMMA => |_: &mut Lexer<R>| Some(Token::Comma),
        LPAREN => |_: &mut Lexer<R>| Some(Token::LParen),
        RPAREN => |_: &mut Lexer<R>| Some(Token::RParen),
        BIT_OR => |_: &mut Lexer<R>| Some(Token::BitOr),
        OR => |_: &mut Lexer<R>| Some(Token::Or),
        AND => |_: &mut Lexer<R>| Some(Token::And),
        GT => |_: &mut Lexer<R>| Some(Token::CmpOp(ir::CmpOp::Gt)),
        LT => |_: &mut Lexer<R>| Some(Token::CmpOp(ir::CmpOp::Lt)),
        GE => |_: &mut Lexer<R>| Some(Token::CmpOp(ir::CmpOp::Geq)),
        LE => |_: &mut Lexer<R>| Some(Token::CmpOp(ir::CmpOp::Leq)),
        EQUALS => |_: &mut Lexer<R>| Some(Token::CmpOp(ir::CmpOp::Eq)),
        NOT_EQUALS => |_: &mut Lexer<R>| Some(Token::CmpOp(ir::CmpOp::Neq)),
        EQUAL => |_: &mut Lexer<R>| Some(Token::Equal),
        DOC => |lex: &mut Lexer<R>| { lex.DOC_MODE(); None },
        END => |_: &mut Lexer<R>| Some(Token::End),
        SYMMETRIC => |_: &mut Lexer<R>| Some(Token::Symmetric),
        ANTISYMMETRIC => |_: &mut Lexer<R>| Some(Token::AntiSymmetric),
        ARROW => |_: &mut Lexer<R>| Some(Token::Arrow),
        DIVIDE => |_: &mut Lexer<R>| Some(Token::Divide),
    }

    DOC_MODE {
        . => |lex: &mut Lexer<R>| {
            let s = lex.yystr();
            lex.comment.push_str(&s);
            None
        },
        WHITESPACE* '\n' => |lex: &mut Lexer<R>| {
            lex.INITIAL();
            Some(Token::Doc(std::mem::replace(&mut lex.comment, String::new())))
        },
        WHITESPACE* '\n' WHITESPACE* DOC => |_: &mut Lexer<R>| None,
    }

    COMMENT_MODE {
        . => |_: &mut Lexer<R>| None,
        C_COMMENT_END => |lex: &mut Lexer<R>| {
            lex.INITIAL();
            None
        }
    }
}
