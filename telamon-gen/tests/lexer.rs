extern crate telamon_gen;
extern crate errno;

use telamon_gen::lexer::*;
use telamon_gen::ir::{CounterKind, CounterVisibility, SetDefKey, CmpOp};

use errno::Errno;

#[test]
fn lexer_initial() {
    // Invalid's Token
    assert_eq!(Lexer::new(b"!".to_vec()).collect::<Vec<_>>(), vec![
                Err(LexicalError::InvalidToken(
                   LexerPosition::default(),
                   Token::InvalidToken(String::from("!")),
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);

    // ChoiceIdent's Token
    assert_eq!(Lexer::new(b"az_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                    Token::ChoiceIdent(String::from("az_09")),
                    LexerPosition::from(Position { column: 5, ..Default::default() }) 
                )),
              ]);
    // SetIdent's Token
    assert_eq!(Lexer::new(b"Az_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetIdent(String::from("Az_09")),
                   LexerPosition::from(Position { column: 5, ..Default::default() }) 
                )),
              ]);
    // ValueIdent's Token
    assert_eq!(Lexer::new(b"AZ_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::ValueIdent(String::from("AZ_09")),
                   LexerPosition::from(Position { column: 5, ..Default::default() }) 
                )),
              ]);
    // Var's Token
    assert_eq!(Lexer::new(b"$vV".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Var(String::from("vV")),
                   LexerPosition::from(Position { column: 3, ..Default::default() }) 
                )),
              ]);
    // Alias's Token
    assert_eq!(Lexer::new(b"alias".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Alias,
                   LexerPosition::from(Position { column: 5, ..Default::default() }) 
                )),
              ]);
    // Counter's Token
    assert_eq!(Lexer::new(b"counter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Counter,
                   LexerPosition::from(Position { column: 7, ..Default::default() }) 
                )),
              ]);
    // Define's Token
    assert_eq!(Lexer::new(b"define".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Define,
                   LexerPosition::from(Position { column: 6, ..Default::default() }) 
                )),
              ]);
    // Enum's Token
    assert_eq!(Lexer::new(b"enum".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Enum,
                   LexerPosition::from(Position { column: 4, ..Default::default() }) 
                )),
              ]);
    // Forall's Token
    assert_eq!(Lexer::new(b"forall".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Forall,
                   LexerPosition::from(Position { column: 6, ..Default::default() }) 
                )),
              ]);
    // In's Token
    assert_eq!(Lexer::new(b"in".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::In,
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Is's Token
    assert_eq!(Lexer::new(b"is".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Is,
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Not's Token
    assert_eq!(Lexer::new(b"not".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Not,
                   LexerPosition::from(Position { column: 3, ..Default::default() }) 
                )),
              ]);
    // Require's Token
    assert_eq!(Lexer::new(b"require".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Require,
                   LexerPosition::from(Position { column: 7, ..Default::default() }) 
                )),
              ]);
    // Mul's CounterKind Token
    assert_eq!(Lexer::new(b"mul".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CounterKind(CounterKind::Mul),
                   LexerPosition::from(Position { column: 3, ..Default::default() }) 
                )),
              ]);
    // Sum's CounterKind Token
    assert_eq!(Lexer::new(b"sum".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CounterKind(CounterKind::Add),
                   LexerPosition::from(Position { column: 3, ..Default::default() }) 
                )),
              ]);
    // Value's Token
    assert_eq!(Lexer::new(b"value".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Value,
                   LexerPosition::from(Position { column: 5, ..Default::default() }) 
                )),
              ]);
    // When's Token
    assert_eq!(Lexer::new(b"when".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::When,
                   LexerPosition::from(Position { column: 4, ..Default::default() }) 
                )),
              ]);
    // Trigger's Token
    assert_eq!(Lexer::new(b"trigger".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Trigger,
                   LexerPosition::from(Position { column: 7, ..Default::default() }) 
                )),
              ]);
    // NoMax's CounterVisibility Token
    assert_eq!(Lexer::new(b"half".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CounterVisibility(CounterVisibility::NoMax),
                   LexerPosition::from(Position { column: 4, ..Default::default() }) 
                )),
              ]);
    // HiddenMax's CounterVisibility Token
    assert_eq!(Lexer::new(b"internal".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CounterVisibility(CounterVisibility::HiddenMax),
                   LexerPosition::from(Position { column: 8, ..Default::default() }) 
                )),
              ]);
    // Base's Token
    assert_eq!(Lexer::new(b"base".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Base,
                   LexerPosition::from(Position { column: 4, ..Default::default() }) 
                )),
              ]);
    // item_type's SetDefKey Token
    assert_eq!(Lexer::new(b"item_type".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::ItemType),
                   LexerPosition::from(Position { column: 9, ..Default::default() }) 
                )),
              ]);
    // NewObjs's SetDefKey Token
    assert_eq!(Lexer::new(b"new_objs".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::NewObjs),
                   LexerPosition::from(Position { column: 8, ..Default::default() }) 
                )),
              ]);
    // IdType's SetDefKey Token
    assert_eq!(Lexer::new(b"id_type".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::IdType),
                   LexerPosition::from(Position { column: 7, ..Default::default() }) 
                )),
              ]);
    // ItemGetter's SetDefKey Token
    assert_eq!(Lexer::new(b"item_getter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::ItemGetter),
                   LexerPosition::from(Position { column: 11, ..Default::default() }) 
                )),
              ]);
    // IdGetter's SetDefKey Token
    assert_eq!(Lexer::new(b"id_getter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::IdGetter),
                   LexerPosition::from(Position { column: 9, ..Default::default() }) 
                )),
              ]);
    // Iter's SetDefKey Token
    assert_eq!(Lexer::new(b"iterator".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::Iter),
                   LexerPosition::from(Position { column: 8, ..Default::default() }) 
                )),
              ]);
    // Prefix's SetDefKey Token
    assert_eq!(Lexer::new(b"var_prefix".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::Prefix),
                   LexerPosition::from(Position { column: 10, ..Default::default() }) 
                )),
              ]);
    // Reverse's SetDefKey Token
    assert_eq!(Lexer::new(b"reverse".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::Reverse),
                   LexerPosition::from(Position { column: 7, ..Default::default() }) 
                )),
              ]);
    // AddToSet's SetDefKey Token
    assert_eq!(Lexer::new(b"add_to_set".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::AddToSet),
                   LexerPosition::from(Position { column: 10, ..Default::default() }) 
                )),
              ]);
    // FromSuperset's SetDefKey Token
    assert_eq!(Lexer::new(b"from_superset".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SetDefKey(SetDefKey::FromSuperset),
                   LexerPosition::from(Position { column: 13, ..Default::default() }) 
                )),
              ]);
    // Set's Token
    assert_eq!(Lexer::new(b"set".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Set,
                   LexerPosition::from(Position { column: 3, ..Default::default() }) 
                )),
              ]);
    // SubsetOf's Token
    assert_eq!(Lexer::new(b"subsetof".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::SubsetOf,
                   LexerPosition::from(Position { column: 8, ..Default::default() }) 
                )),
              ]);
    // Disjoint's Token
    assert_eq!(Lexer::new(b"disjoint".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Disjoint,
                   LexerPosition::from(Position { column: 8, ..Default::default() }) 
                )),
              ]);
    // Quotient's Token
    assert_eq!(Lexer::new(b"quotient".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Quotient,
                   LexerPosition::from(Position { column: 8, ..Default::default() })
                )),
              ]);
    // Of's Token
    assert_eq!(Lexer::new(b"of".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Of,
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // False's Bool Token
    assert_eq!(Lexer::new(b"false".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Bool(false),
                   LexerPosition::from(Position { column: 5, ..Default::default() }) 
                )),
              ]);
    // True's Bool Token
    assert_eq!(Lexer::new(b"true".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Bool(true),
                   LexerPosition::from(Position { column: 4, ..Default::default() }) 
                )),
              ]);
    // Colon's Token
    assert_eq!(Lexer::new(b":".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Colon,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // Comma's Token
    assert_eq!(Lexer::new(b",".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Comma,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // LParen's Token
    assert_eq!(Lexer::new(b"(".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::LParen,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // RParen's Token
    assert_eq!(Lexer::new(b")".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::RParen,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // Bitor's Token
    assert_eq!(Lexer::new(b"|".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::BitOr,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // Or's Token
    assert_eq!(Lexer::new(b"||".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Or,
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // And's Token
    assert_eq!(Lexer::new(b"&&".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::And,
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Gt's CmpOp Token
    assert_eq!(Lexer::new(b">".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CmpOp(CmpOp::Gt),
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // Lt's CmpOp Token
    assert_eq!(Lexer::new(b"<".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CmpOp(CmpOp::Lt),
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // Ge's CmpOp Token
    assert_eq!(Lexer::new(b">=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CmpOp(CmpOp::Geq),
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Le's CmpOp Token
    assert_eq!(Lexer::new(b"<=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CmpOp(CmpOp::Leq),
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Eq's CmpOp Token
    assert_eq!(Lexer::new(b"==".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CmpOp(CmpOp::Eq),
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Neq's CmpOp Token
    assert_eq!(Lexer::new(b"!=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::CmpOp(CmpOp::Neq),
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Equal's Token
    assert_eq!(Lexer::new(b"=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Equal,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // End's Token
    assert_eq!(Lexer::new(b"end".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::End,
                   LexerPosition::from(Position { column: 3, ..Default::default() }) 
                )),
              ]);
    // Symmetric's Token
    assert_eq!(Lexer::new(b"symmetric".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Symmetric,
                   LexerPosition::from(Position { column: 9, ..Default::default() }) 
                )),
              ]);
    // AntiSymmetric's Token
    assert_eq!(Lexer::new(b"antisymmetric".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::AntiSymmetric,
                   LexerPosition::from(Position { column: 13, ..Default::default() }) 
                )),
              ]);
    // Arrow's Token
    assert_eq!(Lexer::new(b"->".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Arrow,
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    // Divide's Token
    assert_eq!(Lexer::new(b"/".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Divide,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
              ]);
    // Integer's Token
    assert_eq!(Lexer::new(b"integer".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::Integer,
                   LexerPosition::from(Position { column: 7, ..Default::default() }) 
                )),
              ]);
}

#[test]
fn lexer_comment_mode() {
    // C_COMMENT's Token
    assert_eq!(Lexer::new(b"/* com */ ".to_vec()).collect::<Vec<_>>(), vec![]);
    assert_eq!(Lexer::new(b"/* com \n com */ ".to_vec()).collect::<Vec<_>>(), vec![]);

    assert_eq!(Lexer::new(b"| /* comment */ |".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::default(),
                   Token::BitOr,
                   LexerPosition::from(Position { column: 1, ..Default::default() }) 
                )),
                Ok((LexerPosition::from(Position { column: 16, ..Default::default() }),
                   Token::BitOr,
                   LexerPosition::from(Position { column: 17, ..Default::default() }) 
                )),
               ]);
    assert_eq!(Lexer::new(b"| /* comment \n comment */ |".to_vec()).collect::<Vec<_>>(),
               vec![
                Ok((LexerPosition::default(),
                   Token::BitOr,
                   LexerPosition::from(Position { column: 1, ..Default::default() })
                )),
                Ok((LexerPosition::from(Position { column: 26, line: 1 }),
                   Token::BitOr,
                   LexerPosition::from(Position { column: 27, line: 1 })
                )),
               ]);
}

#[test]
fn lexer_doc_mode() {
    // Outer Line Doc's Token
    assert_eq!(Lexer::new(b"/// comment ".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 0, ..Default::default() }),
                    Token::Doc(String::from(" comment ")),
                    LexerPosition::from(Position { column: 12, ..Default::default() }) 
                )),
              ]);
    assert_eq!(Lexer::new(b" /// comment ".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 1, ..Default::default() }),
                    Token::Doc(String::from(" comment ")),
                    LexerPosition::from(Position { column: 13, ..Default::default() }) 
                )),
              ]);
    // Outer Line MultiDoc's Token
    assert_eq!(Lexer::new(b"/// comment \n /// comment ".to_vec()).collect::<Vec<_>>(),
               vec![
                Ok((LexerPosition::from(Position { column: 0, ..Default::default() }),
                    Token::Doc(String::from(" comment ")),
                    LexerPosition::from(Position { column: 12, ..Default::default() }),
                )),
                Ok((LexerPosition::from(Position { column: 1, line: 1 }),
                    Token::Doc(String::from(" comment ")),
                    LexerPosition::from(Position { column: 13, line: 1 })
                )),
              ]);
    // Line Comment Doc's Token
    assert_eq!(Lexer::new(b"// comment".to_vec()).collect::<Vec<_>>(), vec![]);
    // Line Comment MultiDoc's Token
    assert_eq!(Lexer::new(b"// comment \n // comment".to_vec()).collect::<Vec<_>>(), vec![]);
}

#[test]
fn lexer_code_mode() {
    // Simple Code's Token
    assert_eq!(Lexer::new(b"\"_\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 1, ..Default::default() }),
                   Token::Code(String::from("_")),
                   LexerPosition::from(Position { column: 2, ..Default::default() }) 
                )),
              ]);
    assert_eq!(Lexer::new(b"\"__\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 1, ..Default::default() }),
                   Token::Code(String::from("__")),
                   LexerPosition::from(Position { column: 3, ..Default::default() })
                )),
              ]);
    // Multiline Code's Token
    assert_eq!(Lexer::new(b"\"_\\\n_\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 1, ..Default::default() }),
                   Token::Code(String::from("__")),
                   LexerPosition::from(Position { column: 2, line: 1 })
                )),
              ]);
    assert_eq!(Lexer::new(b"\"_\\\n       _\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 1, ..Default::default() }),
                   Token::Code(String::from("__")),
                   LexerPosition::from(Position { column: 2, line: 1 })
                )),
              ]);
    assert_eq!(Lexer::new(b"\"_\\\n__\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 1, ..Default::default() }),
                   Token::Code(String::from("___")),
                   LexerPosition::from(Position { column: 3, line: 1 }) 
                )),
              ]);
    // Repetition Code's Token
    assert_eq!(Lexer::new(b"\"_\" \"_\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((LexerPosition::from(Position { column: 1, ..Default::default() }),
                   Token::Code(String::from("__")),
                   LexerPosition::from(Position { column: 6, ..Default::default() }) 
                )),
              ]);
}

#[test]
fn lexer_include() {
   assert_eq!(Lexer::new(b"include \"/dev/unexist\"".to_vec()).collect::<Vec<_>>(),
              vec![
               Err(LexicalError::InvalidInclude(
                  LexerPosition::default(),
                  Token::InvalidInclude(String::from("/dev/unexist"), Errno(2)),
                  LexerPosition {
                      position: Position {
                          column: 22, ..Default::default()
                      },
                      ..Default::default()
                  }))
               ]);
    
   let filename: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/enum_foo.exh");
   let include = format!("include \"{}\"", filename);
   assert_eq!(Lexer::new(include.as_bytes().to_vec()).collect::<Vec<_>>(), vec![
              Ok((LexerPosition::new(Position::new(0, 73), filename.to_owned()),
                  Token::Define,
                  LexerPosition::new(Position::new(1, 79), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 80), filename.to_owned()),
                  Token::Enum,
                  LexerPosition::new(Position::new(1, 84), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 85), filename.to_owned()),
                  Token::ChoiceIdent(String::from("foo")),
                  LexerPosition::new(Position::new(1, 88), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 88), filename.to_owned()),
                  Token::LParen,
                  LexerPosition::new(Position::new(1, 89), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 89), filename.to_owned()),
                  Token::RParen,
                  LexerPosition::new(Position::new(1, 90), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 90), filename.to_owned()),
                  Token::Colon,
                  LexerPosition::new(Position::new(1, 91), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(2, 0), filename.to_owned()),
                  Token::End,
                  LexerPosition::new(Position::new(2, 3), filename.to_owned())))
            ]);

   assert_eq!(Lexer::new(include.as_bytes().to_vec()).collect::<Vec<_>>(), vec![
              Ok((LexerPosition::new(Position::new(0, 73), filename.to_owned()),
                  Token::Define,
                  LexerPosition::new(Position::new(1, 79), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 80), filename.to_owned()),
                  Token::Enum,
                  LexerPosition::new(Position::new(1, 84), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 85), filename.to_owned()),
                  Token::ChoiceIdent(String::from("foo")),
                  LexerPosition::new(Position::new(1, 88), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 88), filename.to_owned()),
                  Token::LParen,
                  LexerPosition::new(Position::new(1, 89), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 89), filename.to_owned()),
                  Token::RParen,
                  LexerPosition::new(Position::new(1, 90), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(1, 90), filename.to_owned()),
                  Token::Colon,
                  LexerPosition::new(Position::new(1, 91), filename.to_owned()))),
              Ok((LexerPosition::new(Position::new(2, 0), filename.to_owned()),
                  Token::End,
                  LexerPosition::new(Position::new(2, 3), filename.to_owned())))
            ]);
}
