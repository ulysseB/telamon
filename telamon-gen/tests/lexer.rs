extern crate telamon_gen;

use telamon_gen::lexer::{Lexer,Token};
use telamon_gen::ir::{CounterKind, CounterVisibility, SetDefKey, CmpOp};

#[test]
fn token() {
    // Blank
    assert_eq!(Lexer::from(b" \0".to_vec()).collect::<Vec<Token>>(), vec![]);
    // Invalid's Token
    assert_eq!(Lexer::from(b"!\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::InvalidToken(String::from("!")),
              ]);
    // ChoiceIdent's Token
    assert_eq!(Lexer::from(b"az_09\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::ChoiceIdent(String::from("az_09"))
              ]);
    // SetIdent's Token
    assert_eq!(Lexer::from(b"Az_09\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetIdent(String::from("Az_09"))
              ]);
    // ValueIdent's Token
    assert_eq!(Lexer::from(b"AZ_09\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::ValueIdent(String::from("AZ_09"))
              ]);
    // Var's Token
    assert_eq!(Lexer::from(b"$vV\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Var(String::from("vV")),
              ]);
    // Code's Token
    assert_eq!(Lexer::from(b"\"ir::...\"\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Code(String::from("ir::...")),
              ]);
    // Alias's Token
    assert_eq!(Lexer::from(b"alias\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Alias
              ]);
    // Counter's Token
    assert_eq!(Lexer::from(b"counter\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Counter
              ]);
    // Define's Token
    assert_eq!(Lexer::from(b"define\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Define
              ]);
    // Enum's Token
    assert_eq!(Lexer::from(b"enum\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Enum
              ]);
    // Forall's Token
    assert_eq!(Lexer::from(b"forall\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Forall
              ]);
    // In's Token
    assert_eq!(Lexer::from(b"in\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::In
              ]);
    // Is's Token
    assert_eq!(Lexer::from(b"is\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Is
              ]);
    // Not's Token
    assert_eq!(Lexer::from(b"not\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Not
              ]);
    // Require's Token
    assert_eq!(Lexer::from(b"require\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Require
              ]);
    // Mul's CounterKind Token
    assert_eq!(Lexer::from(b"mul\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterKind(CounterKind::Mul)
              ]);
    // Sum's CounterKind Token
    assert_eq!(Lexer::from(b"sum\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterKind(CounterKind::Add)
              ]);
    // Value's Token
    assert_eq!(Lexer::from(b"value\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Value
              ]);
    // When's Token
    assert_eq!(Lexer::from(b"when\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::When
              ]);
    // Trigger's Token
    assert_eq!(Lexer::from(b"trigger\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Trigger
              ]);
    // NoMax's CounterVisibility Token
    assert_eq!(Lexer::from(b"half\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterVisibility(CounterVisibility::NoMax)
              ]);
    // HiddenMax's CounterVisibility Token
    assert_eq!(Lexer::from(b"internal\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterVisibility(CounterVisibility::HiddenMax)
              ]);
    // Base's Token
    assert_eq!(Lexer::from(b"base\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Base
              ]);
    // item_type's SetDefKey Token
    assert_eq!(Lexer::from(b"item_type\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::ItemType)
              ]);
    // NewObjs's SetDefKey Token
    assert_eq!(Lexer::from(b"new_objs\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::NewObjs)
              ]);
    // IdType's SetDefKey Token
    assert_eq!(Lexer::from(b"id_type\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::IdType)
              ]);
    // ItemGetter's SetDefKey Token
    assert_eq!(Lexer::from(b"item_getter\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::ItemGetter)
              ]);
    // IdGetter's SetDefKey Token
    assert_eq!(Lexer::from(b"id_getter\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::IdGetter)
              ]);
    // Iter's SetDefKey Token
    assert_eq!(Lexer::from(b"iterator\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::Iter)
              ]);
    // Prefix's SetDefKey Token
    assert_eq!(Lexer::from(b"var_prefix\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::Prefix)
              ]);
    // Reverse's SetDefKey Token
    assert_eq!(Lexer::from(b"reverse\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::Reverse)
              ]);
    // AddToSet's SetDefKey Token
    assert_eq!(Lexer::from(b"add_to_set\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::AddToSet)
              ]);
    // FromSuperset's SetDefKey Token
    assert_eq!(Lexer::from(b"from_superset\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::FromSuperset)
              ]);
    // Set's Token
    assert_eq!(Lexer::from(b"set\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Set
              ]);
    // SubsetOf's Token
    assert_eq!(Lexer::from(b"subsetof\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SubsetOf
              ]);
    // Disjoint's Token
    assert_eq!(Lexer::from(b"disjoint\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Disjoint
              ]);
    // Quotient's Token
    assert_eq!(Lexer::from(b"quotient\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Quotient
              ]);
    // Of's Token
    assert_eq!(Lexer::from(b"of\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Of
              ]);
    // False's Bool Token
    assert_eq!(Lexer::from(b"false\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Bool(false)
              ]);
    // True's Bool Token
    assert_eq!(Lexer::from(b"true\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Bool(true)
              ]);
    // Colon's Token
    assert_eq!(Lexer::from(b":\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Colon
              ]);
    // Comma's Token
    assert_eq!(Lexer::from(b",\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Comma
              ]);
    // LParen's Token
    assert_eq!(Lexer::from(b"(\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::LParen
              ]);
    // RParen's Token
    assert_eq!(Lexer::from(b")\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::RParen
              ]);
    // Bitor's Token
    assert_eq!(Lexer::from(b"|\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::BitOr
              ]);
    // Or's Token
    assert_eq!(Lexer::from(b"||\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Or
              ]);
    // And's Token
    assert_eq!(Lexer::from(b"&&\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::And
              ]);
    // Gt's CmpOp Token
    assert_eq!(Lexer::from(b">\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Gt)
              ]);
    // Lt's CmpOp Token
    assert_eq!(Lexer::from(b"<\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Lt)
              ]);
    // Ge's CmpOp Token
    assert_eq!(Lexer::from(b">=\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Geq)
              ]);
    // Le's CmpOp Token
    assert_eq!(Lexer::from(b"<=\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Leq)
              ]);
    // Eq's CmpOp Token
    assert_eq!(Lexer::from(b"==\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Eq)
              ]);
    // Neq's CmpOp Token
    assert_eq!(Lexer::from(b"!=\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Neq)
              ]);
    // Equal's Token
    assert_eq!(Lexer::from(b"=\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Equal
              ]);
    // End's Token
    assert_eq!(Lexer::from(b"end\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::End
              ]);
    // Symmetric's Token
    assert_eq!(Lexer::from(b"symmetric\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Symmetric
              ]);
    // AntiSymmetric's Token
    assert_eq!(Lexer::from(b"antisymmetric\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::AntiSymmetric
              ]);
    // Arrow's Token
    assert_eq!(Lexer::from(b"->\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Arrow
              ]);
    // Divide's Token
    assert_eq!(Lexer::from(b"/\0".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Divide
              ]);
}
