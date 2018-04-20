extern crate telamon_gen;

use telamon_gen::lexer::{Lexer,Token};

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
