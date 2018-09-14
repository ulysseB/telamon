pub use super::utils::RcStr;

pub use super::telamon_gen::ast::*;
pub use super::telamon_gen::ir;
pub use super::telamon_gen::lexer::{Lexer, LexerPosition, Position, Spanned};
pub use super::telamon_gen::parser;

#[cfg(test)]
mod redefinition {
    pub use super::*;

    /// Redefinition of the Foo Set.
    #[test]
    fn set() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Foo:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
            end
            set Foo:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
            end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::Redefinition {
                object_kind: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition { line: 0, column: 7 },
                        ..Default::default()
                    },
                    data: Hint::Set,
                },
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 9,
                            column: 16
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 9,
                            column: 19
                        },
                        ..Default::default()
                    },
                    data: String::from("Foo"),
                }
            })
        );
    }

    /// Redefinition of the Field from Set.
    #[test]
    fn field() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Foo:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
                new_objs = \"$objs.inst\"
            end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::Redefinition {
                object_kind: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 7,
                            column: 16
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 7,
                            column: 24
                        },
                        ..Default::default()
                    },
                    data: Hint::Set,
                },
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 8,
                            column: 16
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 8,
                            column: 24
                        },
                        ..Default::default()
                    },
                    data: String::from("NewObjs"),
                }
            })
        );
    }
}

/// Undefined
#[cfg(test)]
mod undefined {
    pub use super::*;

    /// Missing the set Instruction from a Set
    #[test]
    fn parameter() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Operand($inst in Instruction):
                item_type = \"ir::operand::Obj\"
                id_type = \"ir::operand::Id\"
                item_getter = \"ir::operand::get($fun, $inst, $id)\"
                id_getter = \"$item.id()\"
                iterator = \"ir::operand::iter($fun, ir::inst::Obj::id($inst))\"
                var_prefix = \"op\"
                new_objs = \"$objs.operand\"
              end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::Undefined {
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 11
                        },
                        ..Default::default()
                    },
                    data: String::from("Instruction"),
                }
            })
        );
    }

    /// Missing the subset BasicBlock from a Set.
    #[test]
    fn subsetof() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Instruction subsetof BasicBlock:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
                from_superset = \"ir::inst::from_superset($fun, $item)\"
             end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::Undefined {
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: String::from("BasicBlock"),
                }
            })
        );
    }

    #[test]
    fn reverse() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Params:
                    item_type = \"ir::param::Obj\"
                    id_type = \"ir::param::Id\"
                    item_getter = \"ir::param::get($fun, $id)\"
                    id_getter = \"ir::param::Obj::id($item)\"
                    iterator = \"ir::param::iter($fun)\"
                    var_prefix = \"param\"
                    new_objs = \"$objs.param\"
                  end

                  set Values:
                    item_type = \"ir::value::Obj\"
                    id_type = \"ir::value::Id\"
                    item_getter = \"ir::value::get($fun, $id)\"
                    id_getter = \"ir::value::Obj::id($item)\"
                    iterator = \"ir::value::iter($fun)\"
                    var_prefix = \"val\"
                    new_objs = \"$objs.value\"
                  end

                  set ValuesAB($param in Params) subsetof Values:
                    item_type = \"ir::value::Obj\"
                    id_type = \"ir::value::Id\"
                    item_getter = \"ir::value::get($fun, $id)\"
                    id_getter = \"ir::value::Obj::id($item)\"
                    iterator = \"ir::value_ab::iter($fun, ir::param::Obj::id($param))\"
                    var_prefix = \"value_ab\"
                    new_objs = \"$objs.value_ab\"
                    from_superset = \"ir::value_ab::from_superset($fun, $param, $item)\"
                    reverse forall $val in Unvalues =
                        \"ir::value_ab::reverse($fun, $val.id())\"
                  end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::Undefined {
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 20,
                            column: 22
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 20,
                            column: 30
                        },
                        ..Default::default()
                    },
                    data: String::from("Unvalues"),
                }
            })
        );
    }

    /// Missing the Instruction Disjoint from a Set.
    #[test]
    fn disjoint() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Dimensions:
                    disjoint: Instructions
                    item_type = \"ir::Dimension\"
                    id_type = \"ir::DimId\"
                    item_getter = \"$fun.dim($id)\"
                    id_getter = \"$item.id()\"
                    iterator = \"$fun.dims()\"
                    var_prefix = \"dim\"
                    from_superset = \"$item.as_dim()\"
                    new_objs = \"$objs.dimensions\"
                  end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::Undefined {
                object_name: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 14
                        },
                        ..Default::default()
                    },
                    data: String::from("Instructions"),
                }
            })
        );
    }
}

/// Missing Entry
#[cfg(test)]
mod missing_entry {
    pub use super::*;

    /// Missing the ItemType's key from Set.
    #[test]
    fn item_type() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Instruction:
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::MissingEntry {
                object_name: String::from("Instruction"),
                object_field: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: ir::SetDefKey::ItemType.to_string(),
                }
            })
        );
    }

    /// Missing the IdType's key from Set.
    #[test]
    fn id_type() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Instruction:
                item_type = \"ir::inst::Obj\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::MissingEntry {
                object_name: String::from("Instruction"),
                object_field: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: ir::SetDefKey::IdType.to_string(),
                }
            })
        );
    }

    /// Missing the ItemGetter's key from Set.
    #[test]
    fn item_getter() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Instruction:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::MissingEntry {
                object_name: String::from("Instruction"),
                object_field: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: ir::SetDefKey::ItemGetter.to_string(),
                }
            })
        );
    }

    /// Missing the IdGetter's key from Set.
    #[test]
    fn id_getter() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Instruction:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::MissingEntry {
                object_name: String::from("Instruction"),
                object_field: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: ir::SetDefKey::IdGetter.to_string(),
                }
            })
        );
    }

    /// Missing the Iter's key from Set.
    #[test]
    fn iter() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Instruction:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
              end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::MissingEntry {
                object_name: String::from("Instruction"),
                object_field: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: ir::SetDefKey::Iter.to_string(),
                }
            })
        );
    }

    /// Missing the NewObjs's key from Set.
    #[test]
    fn new_objs() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set Instruction:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
              end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::MissingEntry {
                object_name: String::from("Instruction"),
                object_field: Spanned {
                    beg: Position {
                        position: LexerPosition { line: 0, column: 4 },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 0,
                            column: 15
                        },
                        ..Default::default()
                    },
                    data: ir::SetDefKey::NewObjs.to_string(),
                }
            })
        );
    }

    /// Missing the SetDefKey's key from Set.
    #[test]
    fn from_superset() {
        assert_eq!(
            parser::parse_ast(Lexer::new(
                b"set BasicBlock:
                item_type = \"ir::basic_block::Obj\"
                id_type = \"ir::basic_block::Id\"
                item_getter = \"ir::basic_block::get($fun, $id)\"
                id_getter = \"ir::basic_block::Obj::id($item)\"
                iterator = \"ir::basic_block::iter($fun)\"
                var_prefix = \"bb\"
                new_objs = \"$objs.basic_block\"
              end

              set Instruction subsetof BasicBlock:
                item_type = \"ir::inst::Obj\"
                id_type = \"ir::inst::Id\"
                item_getter = \"ir::inst::get($fun, $id)\"
                id_getter = \"ir::inst::Obj::id($item)\"
                iterator = \"ir::inst::iter($fun)\"
                var_prefix = \"inst\"
                new_objs = \"$objs.inst\"
             end".to_vec()
            )).unwrap()
            .type_check()
            .err(),
            Some(TypeError::MissingEntry {
                object_name: String::from("Instruction"),
                object_field: Spanned {
                    beg: Position {
                        position: LexerPosition {
                            line: 10,
                            column: 18
                        },
                        ..Default::default()
                    },
                    end: Position {
                        position: LexerPosition {
                            line: 10,
                            column: 29
                        },
                        ..Default::default()
                    },
                    data: ir::SetDefKey::FromSuperset.to_string(),
                }
            })
        );
    }
}
