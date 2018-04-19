extern crate telamon_gen;

use telamon_gen::lexer::Lexer;

#[test]
fn single_enum() {
    let mut ll: Lexer = Lexer::from(
        b"define enum foo():
          end".to_vec());

    while let Some(token) = ll.next() {
        println!("{:?}", token);
    }
}
