extern crate telamon_gen;

use telamon_gen::lexer::{Lexer,Token};
use telamon_gen::ir::{CounterKind, CounterVisibility, SetDefKey, CmpOp};

#[test]
fn initial() {
    // Invalid's Token
    assert_eq!(Lexer::from(b"!".to_vec()).collect::<Vec<Token>>(), vec![
                Token::InvalidToken(String::from("!")),
              ]);
    // ChoiceIdent's Token
    assert_eq!(Lexer::from(b"az_09".to_vec()).collect::<Vec<Token>>(), vec![
                Token::ChoiceIdent(String::from("az_09"))
              ]);
    // SetIdent's Token
    assert_eq!(Lexer::from(b"Az_09".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetIdent(String::from("Az_09"))
              ]);
    // ValueIdent's Token
    assert_eq!(Lexer::from(b"AZ_09".to_vec()).collect::<Vec<Token>>(), vec![
                Token::ValueIdent(String::from("AZ_09"))
              ]);
    // Var's Token
    assert_eq!(Lexer::from(b"$vV".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Var(String::from("vV")),
              ]);
    // Code's Token
    assert_eq!(Lexer::from(b"\"ir::...\"".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Code(String::from("ir::...")),
              ]);
    // Alias's Token
    assert_eq!(Lexer::from(b"alias".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Alias
              ]);
    // Counter's Token
    assert_eq!(Lexer::from(b"counter".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Counter
              ]);
    // Define's Token
    assert_eq!(Lexer::from(b"define".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Define
              ]);
    // Enum's Token
    assert_eq!(Lexer::from(b"enum".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Enum
              ]);
    // Forall's Token
    assert_eq!(Lexer::from(b"forall".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Forall
              ]);
    // In's Token
    assert_eq!(Lexer::from(b"in".to_vec()).collect::<Vec<Token>>(), vec![
                Token::In
              ]);
    // Is's Token
    assert_eq!(Lexer::from(b"is".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Is
              ]);
    // Not's Token
    assert_eq!(Lexer::from(b"not".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Not
              ]);
    // Require's Token
    assert_eq!(Lexer::from(b"require".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Require
              ]);
    // Mul's CounterKind Token
    assert_eq!(Lexer::from(b"mul".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterKind(CounterKind::Mul)
              ]);
    // Sum's CounterKind Token
    assert_eq!(Lexer::from(b"sum".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterKind(CounterKind::Add)
              ]);
    // Value's Token
    assert_eq!(Lexer::from(b"value".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Value
              ]);
    // When's Token
    assert_eq!(Lexer::from(b"when".to_vec()).collect::<Vec<Token>>(), vec![
                Token::When
              ]);
    // Trigger's Token
    assert_eq!(Lexer::from(b"trigger".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Trigger
              ]);
    // NoMax's CounterVisibility Token
    assert_eq!(Lexer::from(b"half".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterVisibility(CounterVisibility::NoMax)
              ]);
    // HiddenMax's CounterVisibility Token
    assert_eq!(Lexer::from(b"internal".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CounterVisibility(CounterVisibility::HiddenMax)
              ]);
    // Base's Token
    assert_eq!(Lexer::from(b"base".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Base
              ]);
    // item_type's SetDefKey Token
    assert_eq!(Lexer::from(b"item_type".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::ItemType)
              ]);
    // NewObjs's SetDefKey Token
    assert_eq!(Lexer::from(b"new_objs".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::NewObjs)
              ]);
    // IdType's SetDefKey Token
    assert_eq!(Lexer::from(b"id_type".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::IdType)
              ]);
    // ItemGetter's SetDefKey Token
    assert_eq!(Lexer::from(b"item_getter".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::ItemGetter)
              ]);
    // IdGetter's SetDefKey Token
    assert_eq!(Lexer::from(b"id_getter".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::IdGetter)
              ]);
    // Iter's SetDefKey Token
    assert_eq!(Lexer::from(b"iterator".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::Iter)
              ]);
    // Prefix's SetDefKey Token
    assert_eq!(Lexer::from(b"var_prefix".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::Prefix)
              ]);
    // Reverse's SetDefKey Token
    assert_eq!(Lexer::from(b"reverse".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::Reverse)
              ]);
    // AddToSet's SetDefKey Token
    assert_eq!(Lexer::from(b"add_to_set".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::AddToSet)
              ]);
    // FromSuperset's SetDefKey Token
    assert_eq!(Lexer::from(b"from_superset".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SetDefKey(SetDefKey::FromSuperset)
              ]);
    // Set's Token
    assert_eq!(Lexer::from(b"set".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Set
              ]);
    // SubsetOf's Token
    assert_eq!(Lexer::from(b"subsetof".to_vec()).collect::<Vec<Token>>(), vec![
                Token::SubsetOf
              ]);
    // Disjoint's Token
    assert_eq!(Lexer::from(b"disjoint".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Disjoint
              ]);
    // Quotient's Token
    assert_eq!(Lexer::from(b"quotient".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Quotient
              ]);
    // Of's Token
    assert_eq!(Lexer::from(b"of".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Of
              ]);
    // False's Bool Token
    assert_eq!(Lexer::from(b"false".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Bool(false)
              ]);
    // True's Bool Token
    assert_eq!(Lexer::from(b"true".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Bool(true)
              ]);
    // Colon's Token
    assert_eq!(Lexer::from(b":".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Colon
              ]);
    // Comma's Token
    assert_eq!(Lexer::from(b",".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Comma
              ]);
    // LParen's Token
    assert_eq!(Lexer::from(b"(".to_vec()).collect::<Vec<Token>>(), vec![
                Token::LParen
              ]);
    // RParen's Token
    assert_eq!(Lexer::from(b")".to_vec()).collect::<Vec<Token>>(), vec![
                Token::RParen
              ]);
    // Bitor's Token
    assert_eq!(Lexer::from(b"|".to_vec()).collect::<Vec<Token>>(), vec![
                Token::BitOr
              ]);
    // Or's Token
    assert_eq!(Lexer::from(b"||".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Or
              ]);
    // And's Token
    assert_eq!(Lexer::from(b"&&".to_vec()).collect::<Vec<Token>>(), vec![
                Token::And
              ]);
    // Gt's CmpOp Token
    assert_eq!(Lexer::from(b">".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Gt)
              ]);
    // Lt's CmpOp Token
    assert_eq!(Lexer::from(b"<".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Lt)
              ]);
    // Ge's CmpOp Token
    assert_eq!(Lexer::from(b">=".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Geq)
              ]);
    // Le's CmpOp Token
    assert_eq!(Lexer::from(b"<=".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Leq)
              ]);
    // Eq's CmpOp Token
    assert_eq!(Lexer::from(b"==".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Eq)
              ]);
    // Neq's CmpOp Token
    assert_eq!(Lexer::from(b"!=".to_vec()).collect::<Vec<Token>>(), vec![
                Token::CmpOp(CmpOp::Neq)
              ]);
    // Equal's Token
    assert_eq!(Lexer::from(b"=".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Equal
              ]);
    // End's Token
    assert_eq!(Lexer::from(b"end".to_vec()).collect::<Vec<Token>>(), vec![
                Token::End
              ]);
    // Symmetric's Token
    assert_eq!(Lexer::from(b"symmetric".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Symmetric
              ]);
    // AntiSymmetric's Token
    assert_eq!(Lexer::from(b"antisymmetric".to_vec()).collect::<Vec<Token>>(), vec![
                Token::AntiSymmetric
              ]);
    // Arrow's Token
    assert_eq!(Lexer::from(b"->".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Arrow
              ]);
    // Divide's Token
    assert_eq!(Lexer::from(b"/".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Divide
              ]);
}

#[test]
fn comment_mode() {
    // C_COMMENT's Token
    assert_eq!(Lexer::from(b"/* comment */ ".to_vec()).collect::<Vec<Token>>(), vec![]);
    assert_eq!(Lexer::from(b"/* comment \n comment */ ".to_vec()).collect::<Vec<Token>>(), vec![]);
}

#[test]
fn doc_mode() {
    // Outer Line Doc's Token
    assert_eq!(Lexer::from(b"/// comment".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Doc(String::from(" comment"))
              ]);
    // Outer Line MultiDoc's Token
    assert_eq!(Lexer::from(b"/// comment \n /// comment".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Doc(String::from(" comment ")),
                Token::Doc(String::from(" comment"))
              ]);
    // Line Comment Doc's Token
    assert_eq!(Lexer::from(b"// comment".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Doc(String::from(" comment"))
              ]);
    // Line Comment MultiDoc's Token
    assert_eq!(Lexer::from(b"// comment \n // comment".to_vec()).collect::<Vec<Token>>(), vec![
                Token::Doc(String::from(" comment ")),
                Token::Doc(String::from(" comment"))
              ]);
}
