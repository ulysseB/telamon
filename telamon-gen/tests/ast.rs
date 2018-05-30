extern crate telamon_gen;
extern crate lalrpop_util;

use telamon_gen::lexer::{Lexer, Token, LexicalError, Position};
use telamon_gen::lexer::*;
use telamon_gen::parser;
use telamon_gen::error;
use telamon_gen::ast;

use lalrpop_util::ParseError;

use std::path::Path;

#[test]
fn duplicata_enum() {
    let ast: ast::Ast =
        parser::parse_ast(Lexer::from(
        b"define enum foo():
          value A:
          value B:
          value C:
          end
          
          define enum foo():
          value A:
          end".to_vec())).unwrap();

    println!("{:?}", ast);
    
    println!("{:?}", ast.type_check().err());
}
