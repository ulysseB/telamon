extern crate telamon_gen;

use telamon_gen::lexer::{Lexer,Token};
use telamon_gen::ir::SetDefKey;

#[test]
fn counter_alloc() {
    let ll: Lexer = Lexer::from(
        b"set Instruction:
            id_type = \"ir::inst::Obj\"
          end".to_vec());

    assert_eq!(ll.collect::<Vec<Token>>(), vec![
                Token::Set,
                Token::SetIdent(String::from("Instruction")),
                Token::Colon,
                Token::SetDefKey(SetDefKey::IdType),
                Token::Code(String::from("ir::inst::Obj")),
                Token::End
              ]);
}

#[test]
fn single_enum() {
    let ll: Lexer = Lexer::from(
        b"define enum foo():
          end".to_vec());

    assert_eq!(ll.collect::<Vec<Token>>(), vec![
                Token::Define,
                Token::Enum,
                Token::ChoiceIdent(String::from("foo")),
                Token::LParen,
                Token::RParen,
                Token::Colon,
                Token::End
              ]);
}
