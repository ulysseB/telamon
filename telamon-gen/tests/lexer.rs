extern crate telamon_gen;

use telamon_gen::lexer::*;
use telamon_gen::ir::{CounterKind, CounterVisibility, SetDefKey, CmpOp};

#[test]
fn initial() {
    // Invalid's Token
    assert_eq!(Lexer::from(b"!".to_vec()).collect::<Vec<_>>(), vec![
                Err(LexicalError::UnexpectedToken(
                   Position::default(),
                   Token::InvalidToken(String::from("!")),
                   Position { column: 1, ..Default::default() } 
                )),
              ]);

    // ChoiceIdent's Token
    assert_eq!(Lexer::from(b"az_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                    Token::ChoiceIdent(String::from("az_09")),
                    Position { column: 5, ..Default::default() } 
                )),
              ]);
    // SetIdent's Token
    assert_eq!(Lexer::from(b"Az_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetIdent(String::from("Az_09")),
                   Position { column: 5, ..Default::default() } 
                )),
              ]);
    // ValueIdent's Token
    assert_eq!(Lexer::from(b"AZ_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::ValueIdent(String::from("AZ_09")),
                   Position { column: 5, ..Default::default() } 
                )),
              ]);
    // Var's Token
    assert_eq!(Lexer::from(b"$vV".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Var(String::from("vV")),
                   Position { column: 3, ..Default::default() } 
                )),
              ]);
    // Code's Token
    assert_eq!(Lexer::from(b"\"ir::...\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Code(String::from("ir::...")),
                   Position { column: 9, ..Default::default() } 
                )),
              ]);
    // Alias's Token
    assert_eq!(Lexer::from(b"alias".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Alias,
                   Position { column: 5, ..Default::default() } 
                )),
              ]);
    // Counter's Token
    assert_eq!(Lexer::from(b"counter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Counter,
                   Position { column: 7, ..Default::default() } 
                )),
              ]);
    // Define's Token
    assert_eq!(Lexer::from(b"define".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Define,
                   Position { column: 6, ..Default::default() } 
                )),
              ]);
    // Enum's Token
    assert_eq!(Lexer::from(b"enum".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Enum,
                   Position { column: 4, ..Default::default() } 
                )),
              ]);
    // Forall's Token
    assert_eq!(Lexer::from(b"forall".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Forall,
                   Position { column: 6, ..Default::default() } 
                )),
              ]);
    // In's Token
    assert_eq!(Lexer::from(b"in".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::In,
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Is's Token
    assert_eq!(Lexer::from(b"is".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Is,
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Not's Token
    assert_eq!(Lexer::from(b"not".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Not,
                   Position { column: 3, ..Default::default() } 
                )),
              ]);
    // Require's Token
    assert_eq!(Lexer::from(b"require".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Require,
                   Position { column: 7, ..Default::default() } 
                )),
              ]);
    // Mul's CounterKind Token
    assert_eq!(Lexer::from(b"mul".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterKind(CounterKind::Mul),
                   Position { column: 3, ..Default::default() } 
                )),
              ]);
    // Sum's CounterKind Token
    assert_eq!(Lexer::from(b"sum".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterKind(CounterKind::Add),
                   Position { column: 3, ..Default::default() } 
                )),
              ]);
    // Value's Token
    assert_eq!(Lexer::from(b"value".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Value,
                   Position { column: 5, ..Default::default() } 
                )),
              ]);
    // When's Token
    assert_eq!(Lexer::from(b"when".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::When,
                   Position { column: 4, ..Default::default() } 
                )),
              ]);
    // Trigger's Token
    assert_eq!(Lexer::from(b"trigger".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Trigger,
                   Position { column: 7, ..Default::default() } 
                )),
              ]);
    // NoMax's CounterVisibility Token
    assert_eq!(Lexer::from(b"half".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterVisibility(CounterVisibility::NoMax),
                   Position { column: 4, ..Default::default() } 
                )),
              ]);
    // HiddenMax's CounterVisibility Token
    assert_eq!(Lexer::from(b"internal".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterVisibility(CounterVisibility::HiddenMax),
                   Position { column: 8, ..Default::default() } 
                )),
              ]);
    // Base's Token
    assert_eq!(Lexer::from(b"base".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Base,
                   Position { column: 4, ..Default::default() } 
                )),
              ]);
    // item_type's SetDefKey Token
    assert_eq!(Lexer::from(b"item_type".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::ItemType),
                   Position { column: 9, ..Default::default() } 
                )),
              ]);
    // NewObjs's SetDefKey Token
    assert_eq!(Lexer::from(b"new_objs".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::NewObjs),
                   Position { column: 8, ..Default::default() } 
                )),
              ]);
    // IdType's SetDefKey Token
    assert_eq!(Lexer::from(b"id_type".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::IdType),
                   Position { column: 7, ..Default::default() } 
                )),
              ]);
    // ItemGetter's SetDefKey Token
    assert_eq!(Lexer::from(b"item_getter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::ItemGetter),
                   Position { column: 11, ..Default::default() } 
                )),
              ]);
    // IdGetter's SetDefKey Token
    assert_eq!(Lexer::from(b"id_getter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::IdGetter),
                   Position { column: 9, ..Default::default() } 
                )),
              ]);
    // Iter's SetDefKey Token
    assert_eq!(Lexer::from(b"iterator".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::Iter),
                   Position { column: 8, ..Default::default() } 
                )),
              ]);
    // Prefix's SetDefKey Token
    assert_eq!(Lexer::from(b"var_prefix".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::Prefix),
                   Position { column: 10, ..Default::default() } 
                )),
              ]);
    // Reverse's SetDefKey Token
    assert_eq!(Lexer::from(b"reverse".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::Reverse),
                   Position { column: 7, ..Default::default() } 
                )),
              ]);
    // AddToSet's SetDefKey Token
    assert_eq!(Lexer::from(b"add_to_set".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::AddToSet),
                   Position { column: 10, ..Default::default() } 
                )),
              ]);
    // FromSuperset's SetDefKey Token
    assert_eq!(Lexer::from(b"from_superset".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::FromSuperset),
                   Position { column: 13, ..Default::default() } 
                )),
              ]);
    // Set's Token
    assert_eq!(Lexer::from(b"set".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Set,
                   Position { column: 3, ..Default::default() } 
                )),
              ]);
    // SubsetOf's Token
    assert_eq!(Lexer::from(b"subsetof".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SubsetOf,
                   Position { column: 8, ..Default::default() } 
                )),
              ]);
    // Disjoint's Token
    assert_eq!(Lexer::from(b"disjoint".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Disjoint,
                   Position { column: 8, ..Default::default() } 
                )),
              ]);
    // Quotient's Token
    assert_eq!(Lexer::from(b"quotient".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Quotient,
                   Position { column: 8, ..Default::default() }
                )),
              ]);
    // Of's Token
    assert_eq!(Lexer::from(b"of".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Of,
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // False's Bool Token
    assert_eq!(Lexer::from(b"false".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Bool(false),
                   Position { column: 5, ..Default::default() } 
                )),
              ]);
    // True's Bool Token
    assert_eq!(Lexer::from(b"true".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Bool(true),
                   Position { column: 4, ..Default::default() } 
                )),
              ]);
    // Colon's Token
    assert_eq!(Lexer::from(b":".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Colon,
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // Comma's Token
    assert_eq!(Lexer::from(b",".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Comma,
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // LParen's Token
    assert_eq!(Lexer::from(b"(".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::LParen,
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // RParen's Token
    assert_eq!(Lexer::from(b")".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::RParen,
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // Bitor's Token
    assert_eq!(Lexer::from(b"|".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::BitOr,
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // Or's Token
    assert_eq!(Lexer::from(b"||".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Or,
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // And's Token
    assert_eq!(Lexer::from(b"&&".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::And,
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Gt's CmpOp Token
    assert_eq!(Lexer::from(b">".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Gt),
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // Lt's CmpOp Token
    assert_eq!(Lexer::from(b"<".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Lt),
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // Ge's CmpOp Token
    assert_eq!(Lexer::from(b">=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Geq),
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Le's CmpOp Token
    assert_eq!(Lexer::from(b"<=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Leq),
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Eq's CmpOp Token
    assert_eq!(Lexer::from(b"==".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Eq),
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Neq's CmpOp Token
    assert_eq!(Lexer::from(b"!=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Neq),
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Equal's Token
    assert_eq!(Lexer::from(b"=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Equal,
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
    // End's Token
    assert_eq!(Lexer::from(b"end".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::End,
                   Position { column: 3, ..Default::default() } 
                )),
              ]);
    // Symmetric's Token
    assert_eq!(Lexer::from(b"symmetric".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Symmetric,
                   Position { column: 9, ..Default::default() } 
                )),
              ]);
    // AntiSymmetric's Token
    assert_eq!(Lexer::from(b"antisymmetric".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::AntiSymmetric,
                   Position { column: 13, ..Default::default() } 
                )),
              ]);
    // Arrow's Token
    assert_eq!(Lexer::from(b"->".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Arrow,
                   Position { column: 2, ..Default::default() } 
                )),
              ]);
    // Divide's Token
    assert_eq!(Lexer::from(b"/".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Divide,
                   Position { column: 1, ..Default::default() } 
                )),
              ]);
}

#[test]
fn comment_mode() {
    // C_COMMENT's Token
    assert_eq!(Lexer::from(b"/* comment */ ".to_vec()).collect::<Vec<_>>(), vec![]);
    assert_eq!(Lexer::from(b"/* comment \n comment */ ".to_vec()).collect::<Vec<_>>(), vec![]);

    assert_eq!(Lexer::from(b"| /* comment */ |".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::BitOr,
                   Position { column: 1, ..Default::default() } 
                )),
                Ok((Position { column: 16, ..Default::default() },
                   Token::BitOr,
                   Position { column: 17, ..Default::default() } 
                )),
               ]);
    assert_eq!(Lexer::from(b"| /* comment \n comment */ |".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::BitOr,
                   Position { column: 1, ..Default::default() } 
                )),
                Ok((Position { column: 26, line: 1 },
                   Token::BitOr,
                   Position { column: 27, line: 1 } 
                )),
               ]);
}

#[test]
fn doc_mode() {
    // Outer Line Doc's Token
    assert_eq!(Lexer::from(b"/// comment ".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { column: 0, ..Default::default() },
                    Token::Doc(String::from(" comment ")),
                    Position { column: 12, ..Default::default() } 
                )),
              ]);
    assert_eq!(Lexer::from(b" /// comment ".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { column: 1, ..Default::default() },
                    Token::Doc(String::from(" comment ")),
                    Position { column: 13, ..Default::default() } 
                )),
              ]);
    // Outer Line MultiDoc's Token
    assert_eq!(Lexer::from(b"/// comment \n /// comment ".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { column: 0, ..Default::default() },
                    Token::Doc(String::from(" comment ")),
                    Position { column: 12, ..Default::default() },
                )),
                Ok((Position { column: 1, line: 1 },
                    Token::Doc(String::from(" comment ")),
                    Position { column: 13, line: 1 }
                )),
              ]);
    // Line Comment Doc's Token
    assert_eq!(Lexer::from(b"// comment".to_vec()).collect::<Vec<_>>(), vec![]);
    // Line Comment MultiDoc's Token
    assert_eq!(Lexer::from(b"// comment \n // comment".to_vec()).collect::<Vec<_>>(), vec![]);
}
