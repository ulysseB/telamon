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
                   Position::default(),
                   Token::InvalidToken(String::from("!")),
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);

    // ChoiceIdent's Token
    assert_eq!(Lexer::new(b"az_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                    Token::ChoiceIdent(String::from("az_09")),
                    Position { position: LexerPosition { column: 5, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // SetIdent's Token
    assert_eq!(Lexer::new(b"Az_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetIdent(String::from("Az_09")),
                   Position { position: LexerPosition { column: 5, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // ValueIdent's Token
    assert_eq!(Lexer::new(b"AZ_09".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::ValueIdent(String::from("AZ_09")),
                   Position { position: LexerPosition { column: 5, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Var's Token
    assert_eq!(Lexer::new(b"$vV".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Var(String::from("vV")),
                   Position { position: LexerPosition { column: 3, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Alias's Token
    assert_eq!(Lexer::new(b"alias".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Alias,
                   Position { position: LexerPosition { column: 5, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Counter's Token
    assert_eq!(Lexer::new(b"counter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Counter,
                   Position { position: LexerPosition { column: 7, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Define's Token
    assert_eq!(Lexer::new(b"define".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Define,
                   Position { position: LexerPosition { column: 6, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Enum's Token
    assert_eq!(Lexer::new(b"enum".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Enum,
                   Position { position: LexerPosition { column: 4, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Forall's Token
    assert_eq!(Lexer::new(b"forall".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Forall,
                   Position { position: LexerPosition { column: 6, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // In's Token
    assert_eq!(Lexer::new(b"in".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::In,
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Is's Token
    assert_eq!(Lexer::new(b"is".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Is,
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Not's Token
    assert_eq!(Lexer::new(b"not".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Not,
                   Position { position: LexerPosition { column: 3, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Require's Token
    assert_eq!(Lexer::new(b"require".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Require,
                   Position { position: LexerPosition { column: 7, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Mul's CounterKind Token
    assert_eq!(Lexer::new(b"mul".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterKind(CounterKind::Mul),
                   Position { position: LexerPosition { column: 3, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Sum's CounterKind Token
    assert_eq!(Lexer::new(b"sum".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterKind(CounterKind::Add),
                   Position { position: LexerPosition { column: 3, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Value's Token
    assert_eq!(Lexer::new(b"value".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Value,
                   Position { position: LexerPosition { column: 5, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // When's Token
    assert_eq!(Lexer::new(b"when".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::When,
                   Position { position: LexerPosition { column: 4, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Trigger's Token
    assert_eq!(Lexer::new(b"trigger".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Trigger,
                   Position { position: LexerPosition { column: 7, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // NoMax's CounterVisibility Token
    assert_eq!(Lexer::new(b"half".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterVisibility(CounterVisibility::NoMax),
                   Position { position: LexerPosition { column: 4, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // HiddenMax's CounterVisibility Token
    assert_eq!(Lexer::new(b"internal".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CounterVisibility(CounterVisibility::HiddenMax),
                   Position { position: LexerPosition { column: 8, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Base's Token
    assert_eq!(Lexer::new(b"base".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Base,
                   Position { position: LexerPosition { column: 4, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // item_type's SetDefKey Token
    assert_eq!(Lexer::new(b"item_type".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::ItemType),
                   Position { position: LexerPosition { column: 9, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // NewObjs's SetDefKey Token
    assert_eq!(Lexer::new(b"new_objs".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::NewObjs),
                   Position { position: LexerPosition { column: 8, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // IdType's SetDefKey Token
    assert_eq!(Lexer::new(b"id_type".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::IdType),
                   Position { position: LexerPosition { column: 7, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // ItemGetter's SetDefKey Token
    assert_eq!(Lexer::new(b"item_getter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::ItemGetter),
                   Position { position: LexerPosition { column: 11, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // IdGetter's SetDefKey Token
    assert_eq!(Lexer::new(b"id_getter".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::IdGetter),
                   Position { position: LexerPosition { column: 9, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Iter's SetDefKey Token
    assert_eq!(Lexer::new(b"iterator".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::Iter),
                   Position { position: LexerPosition { column: 8, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Prefix's SetDefKey Token
    assert_eq!(Lexer::new(b"var_prefix".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::Prefix),
                   Position { position: LexerPosition { column: 10, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Reverse's SetDefKey Token
    assert_eq!(Lexer::new(b"reverse".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::Reverse),
                   Position { position: LexerPosition { column: 7, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // AddToSet's SetDefKey Token
    assert_eq!(Lexer::new(b"add_to_set".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::AddToSet),
                   Position { position: LexerPosition { column: 10, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // FromSuperset's SetDefKey Token
    assert_eq!(Lexer::new(b"from_superset".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SetDefKey(SetDefKey::FromSuperset),
                   Position { position: LexerPosition { column: 13, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Set's Token
    assert_eq!(Lexer::new(b"set".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Set,
                   Position { position: LexerPosition { column: 3, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // SubsetOf's Token
    assert_eq!(Lexer::new(b"subsetof".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::SubsetOf,
                   Position { position: LexerPosition { column: 8, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Disjoint's Token
    assert_eq!(Lexer::new(b"disjoint".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Disjoint,
                   Position { position: LexerPosition { column: 8, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Quotient's Token
    assert_eq!(Lexer::new(b"quotient".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Quotient,
                   Position { position: LexerPosition { column: 8, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Of's Token
    assert_eq!(Lexer::new(b"of".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Of,
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // False's Bool Token
    assert_eq!(Lexer::new(b"false".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Bool(false),
                   Position { position: LexerPosition { column: 5, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // True's Bool Token
    assert_eq!(Lexer::new(b"true".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Bool(true),
                   Position { position: LexerPosition { column: 4, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Colon's Token
    assert_eq!(Lexer::new(b":".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Colon,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Comma's Token
    assert_eq!(Lexer::new(b",".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Comma,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // LParen's Token
    assert_eq!(Lexer::new(b"(".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::LParen,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // RParen's Token
    assert_eq!(Lexer::new(b")".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::RParen,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Bitor's Token
    assert_eq!(Lexer::new(b"|".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::BitOr,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Or's Token
    assert_eq!(Lexer::new(b"||".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Or,
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // And's Token
    assert_eq!(Lexer::new(b"&&".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::And,
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Gt's CmpOp Token
    assert_eq!(Lexer::new(b">".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Gt),
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Lt's CmpOp Token
    assert_eq!(Lexer::new(b"<".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Lt),
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Ge's CmpOp Token
    assert_eq!(Lexer::new(b">=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Geq),
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Le's CmpOp Token
    assert_eq!(Lexer::new(b"<=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Leq),
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Eq's CmpOp Token
    assert_eq!(Lexer::new(b"==".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Eq),
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Neq's CmpOp Token
    assert_eq!(Lexer::new(b"!=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::CmpOp(CmpOp::Neq),
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Equal's Token
    assert_eq!(Lexer::new(b"=".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Equal,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // End's Token
    assert_eq!(Lexer::new(b"end".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::End,
                   Position { position: LexerPosition { column: 3, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Symmetric's Token
    assert_eq!(Lexer::new(b"symmetric".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Symmetric,
                   Position { position: LexerPosition { column: 9, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // AntiSymmetric's Token
    assert_eq!(Lexer::new(b"antisymmetric".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::AntiSymmetric,
                   Position { position: LexerPosition { column: 13, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Arrow's Token
    assert_eq!(Lexer::new(b"->".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Arrow,
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Divide's Token
    assert_eq!(Lexer::new(b"/".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Divide,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Integer's Token
    assert_eq!(Lexer::new(b"integer".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::Integer,
                   Position { position: LexerPosition { column: 7, ..Default::default() }, ..Default::default() }
                )),
              ]);
}

#[test]
fn lexer_comment_mode() {
    // C_COMMENT's Token
    assert_eq!(Lexer::new(b"/* com */ ".to_vec()).collect::<Vec<_>>(), vec![]);
    assert_eq!(Lexer::new(b"/* com \n com */ ".to_vec()).collect::<Vec<_>>(), vec![]);

    assert_eq!(Lexer::new(b"| /* comment */ |".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position::default(),
                   Token::BitOr,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
                Ok((Position { position: LexerPosition { column: 16, ..Default::default() }, ..Default::default() },
                   Token::BitOr,
                   Position { position: LexerPosition { column: 17, ..Default::default() }, ..Default::default() }
                )),
               ]);
    assert_eq!(Lexer::new(b"| /* comment \n comment */ |".to_vec()).collect::<Vec<_>>(),
               vec![
                Ok((Position::default(),
                   Token::BitOr,
                   Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() }
                )),
                Ok((Position { position: LexerPosition { column: 26, line: 1 }, ..Default::default() },
                   Token::BitOr,
                   Position { position: LexerPosition { column: 27, line: 1 }, ..Default::default() }
                )),
               ]);
}

#[test]
fn lexer_doc_mode() {
    // Outer Line Doc's Token
    assert_eq!(Lexer::new(b"/// comment ".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { position: LexerPosition { column: 0, ..Default::default() }, ..Default::default() },
                    Token::Doc(String::from(" comment ")),
                    Position { position: LexerPosition { column: 12, ..Default::default() }, ..Default::default() }
                )),
              ]);
    assert_eq!(Lexer::new(b" /// comment ".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() },
                    Token::Doc(String::from(" comment ")),
                    Position { position: LexerPosition { column: 13, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Outer Line MultiDoc's Token
    assert_eq!(Lexer::new(b"/// comment \n /// comment ".to_vec()).collect::<Vec<_>>(),
               vec![
                Ok((Position { position: LexerPosition { column: 0, ..Default::default() }, ..Default::default() },
                    Token::Doc(String::from(" comment ")),
                    Position { position: LexerPosition { column: 12, ..Default::default() }, ..Default::default() },
                )),
                Ok((Position { position: LexerPosition { column: 1, line: 1 }, ..Default::default() },
                    Token::Doc(String::from(" comment ")),
                    Position { position: LexerPosition { column: 13, line: 1 }, ..Default::default() }
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
                Ok((Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() },
                   Token::Code(String::from("_")),
                   Position { position: LexerPosition { column: 2, ..Default::default() }, ..Default::default() }
                )),
              ]);
    assert_eq!(Lexer::new(b"\"__\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() },
                   Token::Code(String::from("__")),
                   Position { position: LexerPosition { column: 3, ..Default::default() }, ..Default::default() }
                )),
              ]);
    // Multiline Code's Token
    assert_eq!(Lexer::new(b"\"_\\\n_\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() },
                   Token::Code(String::from("__")),
                   Position { position: LexerPosition { column: 2, line: 1 }, ..Default::default() }
                )),
              ]);
    assert_eq!(Lexer::new(b"\"_\\\n       _\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() },
                   Token::Code(String::from("__")),
                   Position { position: LexerPosition { column: 2, line: 1 }, ..Default::default() }
                )),
              ]);
    assert_eq!(Lexer::new(b"\"_\\\n__\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() },
                   Token::Code(String::from("___")),
                   Position { position: LexerPosition { column: 3, line: 1 }, ..Default::default() }
                )),
              ]);
    // Repetition Code's Token
    assert_eq!(Lexer::new(b"\"_\" \"_\"".to_vec()).collect::<Vec<_>>(), vec![
                Ok((Position { position: LexerPosition { column: 1, ..Default::default() }, ..Default::default() },
                   Token::Code(String::from("__")),
                   Position { position: LexerPosition { column: 6, ..Default::default() }, ..Default::default() }
                )),
              ]);
}

#[test]
fn lexer_include() {
   // Unexist include.
   assert_eq!(Lexer::new(b"include \"/dev/unexist\"".to_vec()).collect::<Vec<_>>(),
              vec![
               Err(LexicalError::InvalidInclude(
                  Position::default(),
                  Token::InvalidInclude(String::from("/dev/unexist"), Errno(2)),
                  Position {
                      position: LexerPosition {
                          column: 22, ..Default::default()
                      },
                      ..Default::default()
                  }))
               ]);

   // Header include.
   let filename: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/include_foo.exh");
   let include = format!("include \"{}\"", filename);

   assert_eq!(Lexer::new(include.as_bytes().to_vec()).count(), 9);

   // Two same header include.
   let filename: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/enum_foo.exh");
   let include = format!("include \"{}\"", filename);

   assert_eq!(Lexer::new(include.as_bytes().to_vec()).count(), 7);
   assert_eq!(Lexer::new(include.as_bytes().to_vec()).count(), 7);

   // Sub header include.
   let filename: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/sub/include_foo.exh");
   let include = format!("include \"{}\"", filename);

   assert_eq!(Lexer::new(include.as_bytes().to_vec()).count(), 7);
}

#[test]
fn lexer_include_extra() {
    // Header include.
    let filename: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/foo_bar.exh");
    let include = format!("include \"{}\"", filename);

    assert_eq!(Lexer::new(include.as_bytes().to_vec())
                          .map(|t| t.unwrap())
                          .map(|(_, t, _)| t)
                          .collect::<Vec<_>>(),
               vec![
                   Token::Define, Token::Enum, Token::ChoiceIdent(String::from("foo")), Token::LParen, Token::RParen, Token::Colon,
                   Token::End,
                   Token::Define, Token::Enum, Token::ChoiceIdent(String::from("bar")), Token::LParen, Token::RParen, Token::Colon,
                       Token::Value, Token::ValueIdent(String::from("A")), Token::Colon,
                       Token::Value, Token::ValueIdent(String::from("B")), Token::Colon,
                   Token::End
               ]
    );
}

#[test]
#[ignore]
fn lexer_include_guard() {
   // double header include.
   let filename: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/include_a.exh");
   let include = format!("include \"{}\"", filename);

   let _ = Lexer::new(include.as_bytes().to_vec()).collect::<Vec<_>>();
}
