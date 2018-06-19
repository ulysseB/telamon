#![allow(dead_code, unused_imports)]
extern crate itertools;
#[macro_use]
extern crate telamon_utils as utils;
#[macro_use]
extern crate log;
extern crate env_logger;

fn main() { panic!("use `cargo test` to run tests") }

#[macro_use]
mod ir_gen;

mod fail;

/// Test the type definitions.
mod single_enum {
    define_ir!();
    generated_file!(single_enum);
    use self::single_enum::*;
    use utils::*;

    /// Ensures basic operations on enum domains are working.
    #[test]
    fn enum_operations() {
        let _ = ::env_logger::try_init();

        // Test operators.
        assert_eq!(Foo::A | Foo::B, Foo::AB);
        assert_eq!(Foo::AB & Foo::BC, Foo::B);
        assert_eq!(!Foo::A, Foo::BC);
        // Test modifiers.
        let mut x = Foo::A;
        x.insert(Foo::B);
        assert_eq!(x, Foo::AB);
        x.restrict(Foo::BC);
        assert_eq!(x, Foo::B);
        x |= Foo::AB;
        assert_eq!(x, Foo::AB);
        x &= Foo::BC;
        assert_eq!(x, Foo::B);
        // Test complement.
        let fun = ir::Function::default();
        assert_eq!(Action::Foo(Foo::A).complement(&fun), Some(Action::Foo(Foo::BC)));

        // Ensure `failed`, `all` and `is_failed` are working.
        assert!(Foo::FAILED.is_failed());
        assert!((!Foo::ALL).is_failed());
        // Ensure is_constrained is working.
        assert!(Foo::A.is_constrained());
        assert!(!Foo::AB.is_constrained());
        // Ensure intersects is working.
        assert!(Foo::AB.intersects(Foo::BC));
        assert!(!Foo::A.intersects(Foo::BC));
        // Ensure contains is working.
        assert!(Foo::AB.contains(Foo::FAILED));
        assert!(Foo::AB.contains(Foo::A));
        assert!(!Foo::AB.contains(Foo::BC));
        // Ensure list is working correctly.
        ::itertools::assert_equal(Foo::AB.list(), vec![Foo::A, Foo::B]);
        ::itertools::assert_equal(Foo::ALL.list(), vec![Foo::A, Foo::B, Foo::C]);
        // Ensure the empty filter is working correctly.
        let fun = ir::Function::default();
        let store = DomainStore::default();
        assert_eq!(foo::filter(&fun, &store), Foo::ALL);
    }

    /// Ensures baisc operations on numeric domains are working.
    #[test]
    fn numeric_operation() {
        let _ = ::env_logger::try_init();
        let all0 = NumericSet::all(&[1, 2, 4, 8]);
        let all1 = NumericSet::all(&[2, 3, 4]);
        let inter01 = NumericSet::all(&[2, 4]);
        let empty = NumericSet::all(&[]);

        // Test set operations.
        assert!(all0 != all1);
        assert!(!(all0.contains(all1)));
        assert!(all0.contains(inter01));
        assert!(empty.is_failed());

        let mut restrict = all0;
        restrict.restrict(all1);
        assert_eq!(restrict, inter01);

        let mut all0_2 = all0;
        all0_2.insert(all0);
        assert_eq!(all0_2, all0);

        // Test action complement.
        let fun = ir::Function::default();
        assert_eq!(Action::Bar(NumericSet::all(&[1, 4])).complement(&fun),
                   Some(Action::Bar(NumericSet::all(&[2, 8]))));

        // Test comparison operators.
        assert!(all0.lt(Range::new_eq(&Range::ALL, 9)));
        assert!(!all0.lt(Range::new_eq(&Range::ALL, 8)));
        assert!(all0.gt(Range::new_eq(&Range::ALL, 0)));
        assert!(!all0.gt(Range::new_eq(&Range::ALL, 1)));
        assert!(all0.leq(Range::new_eq(&Range::ALL, 8)));
        assert!(!all0.leq(Range::new_eq(&Range::ALL, 7)));
        assert!(all0.geq(Range::new_eq(&Range::ALL, 1)));
        assert!(!all0.geq(Range::new_eq(&Range::ALL, 2)));

        assert!(all0.neq(NumericSet::all(&[5, 7])));
        assert!(!all0.neq(NumericSet::all(&[4, 7])));

        // Test constructors.
        assert_eq!(NumericSet::new_lt(&[2, 4, 6], 5), NumericSet::all(&[2, 4]));
        assert_eq!(NumericSet::new_gt(&[2, 4, 6], 3), NumericSet::all(&[4, 6]));
    }
}

/// Test each condition separately.
mod single_conditions {
    define_ir! { trait basic_block; struct dim: basic_block; }
    generated_file!(single_conditions);
    use self::single_conditions::*;

    /// Ensures IS and IS NOT conditions are working.
    #[test]
    fn is_conditions() {
        let _ = ::env_logger::try_init();
        let fun = &mut ir::Function::default();
        let bb0_id = ir::dim::create(fun, false).into();
        let bb1_id = ir::dim::create(fun, false).into();
        let mut store = DomainStore::new(&fun);
        let bb0 = ir::basic_block::get(fun, bb0_id);
        // Test IS
        assert_eq!(enum_1::filter(bb0, fun, &store), Enum1::ALL);
        store.set_enum_1(bb1_id, Enum1::A);
        assert_eq!(enum_1::filter(bb0, fun, &store), Enum1::A | Enum1::B);
        store.set_enum_1(bb1_id, Enum1::B);
        assert_eq!(enum_1::filter(bb0, fun, &store), Enum1::ALL);
        store.set_enum_1(bb1_id, Enum1::C);
        assert_eq!(enum_1::filter(bb0, fun, &store), Enum1::B | Enum1::C);
        // Test IS NOT
        assert_eq!(enum_2::filter(bb0, fun, &store), Enum2::ALL);
        store.set_enum_2(bb1_id, Enum2::A);
        assert_eq!(enum_2::filter(bb0, fun, &store), Enum2::A | Enum2::B);
        store.set_enum_2(bb1_id, Enum2::B);
        assert_eq!(enum_2::filter(bb0, fun, &store), Enum2::ALL);
        store.set_enum_2(bb1_id, Enum2::C);
        assert_eq!(enum_2::filter(bb0, fun, &store), Enum2::B | Enum2::C);
    }

    /// Ensures EQUALS and NOT EQUALS conditions are working.
    #[test]
    fn equal_conditions() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let bb0_id = ir::dim::create(fun, false).into();
        let bb1_id = ir::dim::create(fun, false).into();
        let mut store = DomainStore::new(&fun);
        let bb0 = ir::basic_block::get(fun, bb0_id);

        // Test EQUALS.
        assert_eq!(enum_3::filter(bb0, fun, &store), Enum3::ALL);
        store.set_enum_3(bb1_id, Enum3::A);
        assert_eq!(enum_3::filter(bb0, fun, &store), Enum3::A);
        store.set_enum_3(bb1_id, Enum3::B);
        assert_eq!(enum_3::filter(bb0, fun, &store), Enum3::B);
        store.set_enum_3(bb1_id, Enum3::C);
        assert_eq!(enum_3::filter(bb0, fun, &store), Enum3::C);
        // Test NOT EQUALS.
        assert_eq!(enum_4::filter(bb0, fun, &store), Enum4::ALL);
        store.set_enum_4(bb1_id, Enum4::A);
        assert_eq!(enum_4::filter(bb0, fun, &store), Enum4::B | Enum4::C);
        store.set_enum_4(bb1_id, Enum4::B);
        assert_eq!(enum_4::filter(bb0, fun, &store), Enum4::A | Enum4::C);
        store.set_enum_4(bb1_id, Enum4::C);
        assert_eq!(enum_4::filter(bb0, fun, &store), Enum4::A | Enum4::B);
    }

    /// Ensures code conditions are working.
    #[test]
    fn static_code_conditions() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let bb0_id = ir::dim::create(fun, false).into();
        let bb1_id = ir::dim::create(fun, true).into();
        let store = DomainStore::new(&fun);
        let bb0 = ir::basic_block::get(fun, bb0_id);
        let bb1 = ir::basic_block::get(fun, bb1_id);

        assert_eq!(enum_5::filter(bb0, &fun, &store), Enum5::B | Enum5::C);
        assert_eq!(enum_5::filter(bb1, &fun, &store), Enum5::A | Enum5::C);
    }
}

/// Test the combination multiple conditions into a single rule.
mod multiple_conditions {
    define_ir! { trait basic_block; struct inst: basic_block; struct dim: basic_block; }
    generated_file!(multiple_conditions);
    use self::multiple_conditions::*;

    /// Ensures conditions work correctly when referencing multiple enums.
    #[test]
    fn multiple_enums() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let bb0_id = ir::dim::create(fun, false).into();
        let bb1_id = ir::dim::create(fun, false).into();
        let mut store = DomainStore::new(&fun);
        let bb0 = ir::basic_block::get(fun, bb0_id);
        // Test all values of enum2(bb0) for enum1(block1) == A
        store.set_enum_1(bb1_id, Enum1::A);
        store.set_enum_2(bb0_id, Enum2::A);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::A);
        store.set_enum_2(bb0_id, Enum2::B);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::ALL);
        store.set_enum_2(bb0_id, Enum2::C);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::A);
        // Test all values of enum2(bb0) for enum1(block1) == B
        store.set_enum_1(bb1_id, Enum1::B);
        store.set_enum_2(bb0_id, Enum2::A);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::A);
        store.set_enum_2(bb0_id, Enum2::B);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::ALL);
        store.set_enum_2(bb0_id, Enum2::C);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::A);
        // Test all values of enum2(bb0) for enum1(block1) == C
        store.set_enum_1(bb1_id, Enum1::C);
        store.set_enum_2(bb0_id, Enum2::A);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::ALL);
        store.set_enum_2(bb0_id, Enum2::B);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::ALL);
        store.set_enum_2(bb0_id, Enum2::C);
        assert_eq!(enum_1::filter(bb0, &fun, &store), Enum1::ALL);
    }

    /// Ensures we can write conditions with multiple cases on the same input.
    #[test]
    fn mutiple_cases_constraints() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let bb0_id = ir::dim::create(fun, false).into();
        let bb1_id = ir::dim::create(fun, false).into();
        let mut store = DomainStore::new(&fun);
        let bb0 = ir::basic_block::get(fun, bb0_id);
        assert_eq!(enum_3::filter(bb0, &fun, &store), Enum3::ALL);
        store.set_enum_3(bb1_id, Enum3::A);
        assert_eq!(enum_3::filter(bb0, &fun, &store), Enum3::ALL);
        store.set_enum_3(bb1_id, Enum3::B);
        assert_eq!(enum_3::filter(bb0, &fun, &store), Enum3::A | Enum3::C);
        store.set_enum_3(bb1_id, Enum3::C);
        assert_eq!(enum_3::filter(bb0, &fun, &store), Enum3::A | Enum3::B);
    }

    /// Ensures static rules are correctly generated when mixed with conditions on enums.
    #[test]
    fn mixed_code_conditions() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let bb0_id = ir::dim::create(fun, false).into();
        let bb1_id = ir::dim::create(fun, true).into();
        let mut store = DomainStore::new(&fun);
        let bb0 = ir::basic_block::get(fun, bb0_id);
        let bb1 = ir::basic_block::get(fun, bb1_id);
        // Test enum_4 for bb0.
        assert_eq!(enum_4::filter(bb0, &fun, &store), Enum4::ALL);
        store.set_enum_4(bb1_id, Enum4::A);
        assert_eq!(enum_4::filter(bb0, &fun, &store), Enum4::B);
        store.set_enum_4(bb1_id, Enum4::B);
        assert_eq!(enum_4::filter(bb0, &fun, &store), Enum4::ALL);
        store.set_enum_4(bb1_id, Enum4::C);
        assert_eq!(enum_4::filter(bb0, &fun, &store), Enum4::ALL);
        // Test enum_4 for bb1.
        assert_eq!(enum_4::filter(bb1, &fun, &store), Enum4::ALL);
        store.set_enum_4(bb0_id, Enum4::A);
        assert_eq!(enum_4::filter(bb1, &fun, &store), Enum4::B | Enum4::C);
        store.set_enum_4(bb0_id, Enum4::B);
        assert_eq!(enum_4::filter(bb1, &fun, &store), Enum4::ALL);
        store.set_enum_4(bb0_id, Enum4::C);
        assert_eq!(enum_4::filter(bb1, &fun, &store), Enum4::B | Enum4::C);
    }

    /// Ensures subtype conditions are correctly generated.
    #[test]
    fn subtype_conditions() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let dim = ir::dim::create(fun, false);
        let inst = ir::inst::create(fun, false);
        let mut store = DomainStore::new(&fun);
        let dim = ir::dim::get(fun, dim);
        let inst = ir::inst::get(fun, inst);
        // Test subtype conditions within rules.
        assert_eq!(enum_5::filter(dim, &fun, &store), Enum5::A | Enum5::B);
        assert_eq!(enum_5::filter(inst, &fun, &store), Enum5::ALL);
        // Test subtype conditions within filters.
        store.set_enum_6(inst.id(), Enum6::A);
        assert_eq!(enum_5::filter(inst, &fun, &store), Enum5::A);
        store.set_enum_6(inst.id(), Enum6::B);
        assert_eq!(enum_5::filter(inst, &fun, &store), Enum5::ALL);
    }
}

/// Tests the strength of merged filters.
mod filter_strength {
    define_ir! { trait basic_block; struct dim: basic_block; }
    generated_file!(filter_strength);
    use self::filter_strength::*;

    /// Ensures a merge filter can be strictly more powerful than two separate filters.
    #[test]
    fn merged_filter_strength() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let bb0_id = ir::dim::create(fun, false).into();
        let bb1_id = ir::dim::create(fun, false).into();
        let mut store = DomainStore::new(fun);
        let bb0 = ir::basic_block::get(fun, bb0_id);
        assert_eq!(enum_1::filter(bb0, fun, &store), Enum1::ALL);
        store.set_enum_1(bb1_id, Enum1::A | Enum1::B);
        assert_eq!(enum_1::filter(bb0, fun, &store), Enum1::B | Enum1::C);
    }
}

/// Tests the use of symmetry indications.
mod symmetry {
    define_ir! { trait basic_block; struct dim: basic_block; }
    generated_file!(symmetry);
    use self::symmetry::*;

    /// Ensures the inverse function works correctly.
    #[test]
    fn inverse() {
        let _ = ::env_logger::try_init();

        assert_eq!(Enum2::ALL.inverse(), Enum2::ALL);
        assert_eq!(Enum2::FAILED.inverse(), Enum2::FAILED);
        assert_eq!(Enum2::BEFORE.inverse(), Enum2::AFTER);
        assert_eq!(Enum2::AFTER.inverse(), Enum2::BEFORE);
        assert_eq!(Enum2::INNER.inverse(), Enum2::OUTER);
        assert_eq!(Enum2::OUTER.inverse(), Enum2::INNER);
        assert_eq!(Enum2::MERGED.inverse(), Enum2::MERGED);
    }

    /// Ensures (anti)symmetric generators, getters and setters work correctly.
    #[test]
    fn symmetric_store() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let block0 = ir::dim::create(fun, false).into();
        let block1 = ir::dim::create(fun, false).into();
        let block2 = ir::dim::create(fun, false).into();
        let mut store = DomainStore::new(fun);
        // Test symmetric getters.
        store.set_enum_1(block0, block1, Enum1::A);
        assert_eq!(store.get_enum_1(block0, block1), Enum1::A);
        assert_eq!(store.get_enum_1(block1, block0), Enum1::A);
        assert_eq!(store.get_enum_1(block0, block2), Enum1::ALL);
        assert_eq!(store.get_enum_1(block1, block2), Enum1::ALL);
        // Test antisymmetric getters.
        store.set_enum_2(block0, block1, Enum2::BEFORE);
        assert_eq!(store.get_enum_2(block0, block1), Enum2::BEFORE);
        assert_eq!(store.get_enum_2(block1, block0), Enum2::AFTER);
        assert_eq!(store.get_enum_2(block0, block2), Enum2::ALL);
        assert_eq!(store.get_enum_2(block1, block2), Enum2::ALL);
    }

    /// Tests the symmetric normalization.
    #[test]
    fn symmetric_normalization() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let b0_id = ir::dim::create(fun, false).into();
        let b1_id = ir::dim::create(fun, false).into();
        let store = DomainStore::new(&fun);
        let b0 = ir::basic_block::get(fun, b0_id);
        let b1 = ir::basic_block::get(fun, b1_id);

        assert_eq!(enum_3::filter(b0, b1, &fun, &store), Enum3::A | Enum3::B);
        assert_eq!(enum_4::filter(b0, b1, &fun, &store), Enum4::B);
    }

    /// Tests the antisymmetric normalization.
    #[test]
    fn antisymmetric_normalization() {
        let _ = ::env_logger::try_init();

        let fun = &mut ir::Function::default();
        let b0_id = ir::dim::create(fun, false).into();
        let b1_id = ir::dim::create(fun, false).into();
        let store = DomainStore::new(&fun);
        let b0 = ir::basic_block::get(fun, b0_id);
        let b1 = ir::basic_block::get(fun, b1_id);

        assert_eq!(enum_5::filter(b0, b1, &fun, &store), Enum5::MERGED);
        assert_eq!(enum_6::filter(b0, b1, &fun, &store), Enum6::MERGED);
    }
}

/// Tests the the right actions are taken after a domain is restricted..
mod on_change {
    define_ir! { trait basic_block; struct inst: basic_block; struct dim: basic_block; }
    generated_file!(on_change);
    use self::on_change::*;
    use std::sync::Arc;

    /// Ensures `on_change` generates the correct actions.
    #[test]
    fn on_change_simple() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let b0 = ir::dim::create(&mut fun, false).into();
        let b1 = ir::dim::create(&mut fun, false).into();
        let store = &mut DomainStore::new(&fun);
        let fun = &mut Arc::new(fun);
        let (all0, all1) = (Simple0::ALL, Simple1::ALL);
        let diff = &mut DomainDiff::default();

        store.set_simple_0(b0, Simple0::B);
        assert!(simple_0::on_change(all0, Simple0::B, b0, fun, store, diff).is_ok());
        assert_eq!(diff.pop_simple_1_diff(), Some(((b0,), all1, Simple1::A)));
        assert!(diff.is_empty());

        store.set_simple_1(b0, Simple1::A);
        assert!(simple_1::on_change(all1, Simple1::A, b0, fun, store, diff).is_ok());
        assert!(diff.is_empty());

        store.set_simple_0(b1, Simple0::B);
        store.set_simple_1(b1, Simple1::B);
        assert!(simple_0::on_change(all0, Simple0::A, b1, fun, store, diff).is_err());
        assert!(simple_1::on_change(all1, Simple1::A, b1, fun, store, diff).is_err());
    }

    /// Ensures `on_change` works correctly when a forall variable is mapped to an argument.
    #[test]
    fn on_change_forall() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let b0 = ir::dim::create(&mut fun, false).into();
        let b1 = ir::dim::create(&mut fun, false).into();
        let store = &mut DomainStore::new(&fun);
        let fun = &mut Arc::new(fun);
        let all = Forall0::ALL;
        let diff = &mut DomainDiff::default();

        store.set_forall_0(b0, Forall0::A);
        assert!(forall_0::on_change(all, Forall0::A, b0, fun, store, diff).is_ok());
        assert_eq!(diff.pop_forall_0_diff(), Some(((b1,), all, Forall0::A)));
        assert!(diff.is_empty());

        assert!(forall_0::on_change(all, Forall0::A, b1, fun, store, diff).is_ok());
        assert!(diff.is_empty());
    }

    /// Ensures `on_change` works correctly in presence of type constraints.
    #[test]
    fn on_change_type_constraint() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let inst0 = ir::inst::create(&mut fun, false).into();
        let fun = &mut Arc::new(fun);

        // Check with implicit type constraint on the argument
        let diff = &mut DomainDiff::default();
        let store0 = &mut DomainStore::new(&fun);
        store0.set_type_0(dim0.into(), Type0::B);
        assert!(type_0::on_change(
                Type0::ALL, Type0::B, dim0.into(), fun, store0, diff).is_ok());
        assert_eq!(diff.pop_type_1_diff(), Some(((dim0,), Type1::ALL, Type1::B)));
        assert!(diff.is_empty());


        store0.set_type_0(inst0, Type0::B);
        assert!(type_0::on_change(Type0::ALL, Type0::B, inst0, fun, store0, diff).is_ok());
        assert!(diff.is_empty());

        // Check with explicit type constraints.
        let store1 = &mut DomainStore::new(&fun);
        store1.set_type_1(dim0, Type1::A);
        assert!(type_1::on_change(Type1::ALL, Type1::A, dim0, fun, store1, diff).is_ok());
        assert_eq!(diff.pop_type_0_diff(), Some(((dim0.into(),), Type0::ALL, Type0::A)));
        assert!(diff.is_empty());

        // Check with implicit type constraints on foralls.
        let store2 = &mut DomainStore::new(&fun);
        store2.set_type_2(dim0.into(), Type2::B);
        assert!(type_2::on_change(
                Type2::ALL, Type2::A, dim0.into(), fun, store2, diff).is_ok());
        assert_eq!(diff.pop_type_2_diff(), Some(((inst0,), Type2::ALL, Type2::B)));
        assert!(diff.is_empty());
    }
}

#[allow(unused_variables)]
mod counter_def {
    define_ir! { trait basic_block; struct inst: basic_block; struct dim: basic_block; }
    generated_file!(counter_def);
    use self::counter_def::*;
    use std::sync::Arc;

    /// Ensures a counter counts the number of objects correctly.
    #[test]
    fn simple_counter() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);
        let inst0 = ir::inst::create(&mut fun, false);
        let inst1 = ir::inst::create(&mut fun, false);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store,&mut  fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_simple_counter(inst0), Range { min: 5, max: 5 });
        assert_eq!(store.get_simple_counter(inst1), Range { min: 5, max: 5 });
        assert_eq!(store.get_increment_simple_counter(inst0, dim0), Bool::TRUE);
        assert_eq!(store.get_increment_simple_counter(inst0, dim1), Bool::TRUE);
    }

    /// Tests a counter with a single condition on `incr`.
    #[test]
    fn counter_single_cond() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);
        let dim2 = ir::dim::create(&mut fun, false);
        let dim3 = ir::dim::create(&mut fun, false);
        let _dim4 = ir::dim::create(&mut fun, false);
        let _dim5 = ir::dim::create(&mut fun, false);
        let inst0 = ir::inst::create(&mut fun, false);
        let store = &mut DomainStore::new(&fun);
        store.set_foo(dim0, Foo::A);
        store.set_foo(dim1, Foo::B);
        // Test initialization
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_counter_single_cond(inst0), Range { min: 1, max: 5 });
        // Check if the counter is updated when the conditions are updated
        let actions = vec![Action::Foo(dim2, Foo::A), Action::Foo(dim3, Foo::B)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_counter_single_cond(inst0), Range { min: 2, max: 4 });
    }

    /// Test a counter of counter.
    #[test]
    fn counter_of_counter() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);
        let dim2 = ir::dim::create(&mut fun, false);
        let inst0 = ir::inst::create(&mut fun, false);
        let _inst1 = ir::inst::create(&mut fun, false);
        let store = &mut DomainStore::new(&fun);
        // Test the counter is correctly computed on initialization.
        store.set_foo(dim0, Foo::A);
        store.set_bar(inst0, Bar::A);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_counter_of_counter(dim2.into()), Range { min: 2, max: 8 });
        // Test the counter is correctly computed after the value is updated.
        assert!(apply_decisions(vec![Action::Foo(dim1, Foo::B)], fun, store).is_ok());
        assert_eq!(store.get_counter_of_counter(dim2.into()), Range {min: 2, max: 4 });
        // Test the counter value is correctly restricted.
        let diff = &mut DomainDiff::default();
        assert!(counter_of_counter::restrict(
                dim1.into(), fun, store, Range { min: 2, max: 2 }, diff).is_ok());
        assert_eq!(store.get_foo(dim1.into()), Foo::B);
    }

    /// Test a counter based on a symmetric decision.
    #[test]
    fn counter_of_symmetric() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());

        // Test the counter value at initialization.
        assert_eq!(store.get_counter_of_symm(dim0), Range { min: 0, max: 1 });
        assert_eq!(store.get_counter_of_symm(dim1), Range { min: 0, max: 1 });
        // Test the counter value after a decision is restricted.
        let actions = vec![Action::SymmEnum(dim0, dim1, SymmEnum::A)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_counter_of_symm(dim0), Range { min: 1, max: 1 });
        assert_eq!(store.get_counter_of_symm(dim1), Range { min: 1, max: 1 });
    }
}

mod counter_alloc {
    define_ir! { struct inst; struct dim; }
    generated_file!(counter_alloc);
    use self::counter_alloc::*;
    use std::sync::Arc;

    /// Ensures counters and increments are correctly handled when a new object is allocated.
    #[test]
    fn counter_allocation() {
        let _ = ::env_logger::try_init();
        use counter_alloc::*;

        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let inst0 = ir::inst::create(&mut fun, false);

        let fun = &mut Arc::new(fun);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, Arc::make_mut(fun)).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());

        // Allocate a dimension.
        let dim1 = ir::dim::create(Arc::make_mut(fun), false);
        let new_objs = ir::NewObjs { dim: vec![dim1], .. Default::default() };
        store.alloc(fun, &new_objs);
        let actions = init_domain_partial(
            store, Arc::make_mut(fun), &new_objs, &mut DomainDiff::default()).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_num_foo_a(inst0), Range { min: 0, max: 1 });
        assert_eq!(store.get_foo(inst0, dim0), Foo::ALL);
        assert_eq!(store.get_foo(inst0, dim1), Foo::B);

        // Allocate an instruction
        let inst1 = ir::inst::create(Arc::make_mut(fun), false);
        let new_objs = ir::NewObjs { inst: vec![inst1], .. Default::default() };
        store.alloc(fun, &new_objs);
        let actions = init_domain_partial(
            store, Arc::make_mut(fun), &new_objs, &mut DomainDiff::default()).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_num_foo_a(inst1), Range { min: 0, max: 2 });
        assert_eq!(store.get_foo(inst1, dim0), Foo::ALL);
    }

    /// Ensures internal counters are correctly updated after an allocation.
    #[test]
    fn internal() {
        let _ = ::env_logger::try_init();
        // Initialize the ir_instance and domain.
        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let inst0 = ir::inst::create(&mut fun, false);
        let fun = &mut Arc::new(fun);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, Arc::make_mut(fun)).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());

        // Set the (inst0, dim0) increment and propagate to the counter, but not the
        // counter modification itself.
        let diff = &mut DomainDiff::default();
        store.set_bar(inst0, dim0, Bar::B);
        assert!(store.restrict_num_bar(inst0, Range { min: 0, max: 1}, diff).is_ok());
        let expected_diff = (Range { min: 0, max: 1 }, Range { min: 0, max: 0});
        assert_eq!(diff.num_bar.get(&(inst0,)), Some(&expected_diff));
        // Allocate three dimensions.
        let dim1 = ir::dim::create(Arc::make_mut(fun), false);
        let dim2 = ir::dim::create(Arc::make_mut(fun), false);
        let dim3 = ir::dim::create(Arc::make_mut(fun), false);
        let ref new_objs = ir::NewObjs { dim: vec![dim1, dim2, dim3], inst: vec![] };
        store.alloc(fun, new_objs);
        // Set the (inst0, dim1) increment to false and (inst0, dim2) increment to true.
        store.set_bar(inst0, dim1, Bar::B);
        store.set_bar(inst0, dim2, Bar::A);
        // Re-init the domain and check the diff has been updated.
        let actions = init_domain_partial(store, Arc::make_mut(fun), new_objs, diff);
        assert_eq!(store.get_num_bar(inst0), Range { min: 0, max: 3});
        assert_eq!(diff.num_bar.get(&(inst0,)), None);
        // Propagate all changes and check the final counter value
        for action in actions.unwrap() {
            assert!(apply_action(action, store, diff).is_ok());
        }
        while !diff.is_empty() { assert!(propagate_changes(diff, fun, store).is_ok()); }
        assert_eq!(store.get_num_bar(inst0), Range { min: 1, max: 2 });
    }
}

mod counter_cond {
    define_ir! { struct dim; }
    generated_file!(counter_cond);
    use self::counter_cond::*;
    use std::sync::Arc;

    /// Test counter conditions.
    #[test]
    fn counter_conditions() {
        let _ = ::env_logger::try_init();
        use counter_cond::*;

        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);
        let dim2 = ir::dim::create(&mut fun, false);
        // Test the lower-bound condition.
        let mut store = DomainStore::new(&fun);
        let actions = init_domain(&mut store, &mut fun).unwrap();
        let mut fun = Arc::new(fun);
        assert!(apply_decisions(actions, &mut fun, &mut store).is_ok());
        assert_eq!(store.get_foo(dim0), Foo::ALL);
        let actions = vec![Action::Foo(dim1, Foo::B), Action::Foo(dim2, Foo::B)];
        assert!(apply_decisions(actions, &mut fun, &mut store).is_ok());
        assert_eq!(store.get_foo(dim0), Foo::A);
        // Test the upper bound condition.
        let mut store = DomainStore::new(&fun);
        let actions = init_domain(&mut store, Arc::make_mut(&mut fun)).unwrap();
        assert!(apply_decisions(actions, &mut fun, &mut store).is_ok());
        assert_eq!(store.get_foo(dim0), Foo::ALL);
        let actions = vec![Action::Foo(dim1, Foo::A), Action::Foo(dim2, Foo::A)];
        assert!(apply_decisions(actions, &mut fun, &mut store).is_ok());
        assert_eq!(store.get_foo(dim0), Foo::B);
    }
}

mod antisymmetric_increment {
    define_ir! { struct set_0; }
    generated_file!(antisymmetric_increment);
    use self::antisymmetric_increment::*;
    use std::sync::Arc;

    /// Ensure counters are working when the increment is antisymmetric.
    #[test]
    fn antisymmetric_increment() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let item0 = ir::set_0::create(&mut fun, false);
        let item1 = ir::set_0::create(&mut fun, false);
        let mut store = DomainStore::new(&fun);
        let actions = init_domain(&mut store, &mut fun).unwrap();
        let mut fun = Arc::new(fun);
        assert!(apply_decisions(actions, &mut fun, &mut store).is_ok());
        assert_eq!(store.get_bar(item0), Range { min: 0, max: 1 });
        assert_eq!(store.get_bar(item1), Range { min: 0, max: 1 });

        let actions = vec![Action::Foo(item0, item1, Foo::A)];
        assert!(apply_decisions(actions, &mut fun, &mut store).is_ok());
        assert_eq!(store.get_bar(item0), Range { min: 1, max: 1 });
        assert_eq!(store.get_bar(item1), Range { min: 0, max: 0 });
    }
}

#[allow(unused_variables)]
mod half_counter {
    define_ir! { struct inst; struct dim; }
    generated_file!(half_counter);
}

mod lowering {
    define_ir! { trait basic_block; struct inst: basic_block; struct dim: basic_block; }
    generated_file!(lowering);
    use self::lowering::*;
    use std::sync::Arc;

    fn test_trigger(fun: &mut ir::Function, _: ir::dim::Id, bb: ir::basic_block::Id)
            -> Result<(ir::NewObjs, Vec<Action>), ()> {
        ir::basic_block::get(fun, bb).set_condition();
        Ok(Default::default())
    }

    fn test_complex_trigger(fun: &mut ir::Function, _: ir::dim::Id, dim: ir::dim::Id)
        -> Result<(ir::NewObjs, Vec<Action>), ()>
    {
        ir::dim::get(fun, dim).set_condition();
        Ok(Default::default())
    }

    /// Ensures triggers are correctly called.
    #[test]
    fn trigger() {
        let _ = ::env_logger::try_init();

        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let inst1 = ir::inst::create(&mut fun, false);
        let inst2 = ir::inst::create(&mut fun, false);
        let inst3 = ir::inst::create(&mut fun, false);
        let store = &mut DomainStore::new(&fun);

        // Test trigger on initialization.
        let fun = &mut Arc::new(fun);
        store.set_foo(dim0.into(), inst1.into(), Foo::A);
        let actions = init_domain(store, Arc::make_mut(fun)).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert!(ir::inst::get(fun, inst1).condition());
        assert!(!ir::inst::get(fun, inst2).condition());
        // Test trigger on propagation
        let action = Action::Foo(dim0.into(), inst2.into(), Foo::A);
        assert!(apply_decisions(vec![action], fun, store).is_ok());
        assert!(ir::inst::get(fun, inst2).condition());
        assert!(!ir::inst::get(fun, inst3).condition());
        // Test trigger on alloc
        let inst4 = ir::inst::create(Arc::make_mut(fun), false);
        let new_objs = ir::NewObjs {
            inst: vec![inst4],
            basic_block: vec![inst4.into()],
            .. Default::default()
        };
        store.alloc(fun, &new_objs);
        store.set_foo(dim0.into(), inst4.into(), Foo::A);
        let actions = init_domain_partial(
            store, Arc::make_mut(fun), &new_objs, &mut DomainDiff::default()).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert!(ir::inst::get(fun, inst4).condition());
        assert!(!ir::inst::get(fun, inst3).condition());
        // Ensure type constraints are respected
        assert!(!ir::dim::get(fun, dim0).condition());
    }

    /// Test triggers involving multiple choices.
    #[test]
    fn complex_trigger() {
        let mut fun = ir::Function::default();
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);
        let dim2 = ir::dim::create(&mut fun, false);
        let store = &mut DomainStore::new(&fun);
        let fun = &mut Arc::new(fun);
        let actions = init_domain(store, Arc::make_mut(fun)).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());

        // Test the triggers are called when decisions are set.
        let actions = vec![Action::Bar(dim0, Bar::A), Action::Bar(dim1, Bar::A)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert!(ir::dim::get(fun, dim0).condition());
        assert!(ir::dim::get(fun, dim1).condition());
        assert!(!ir::dim::get(fun, dim2).condition());
        // Test triggers are correctly called during partial initialization, but not
        // when a triggerring value is still not propagated.
        let diff = &mut DomainDiff::default();
        assert!(store.restrict_bar(dim2, Bar::A, diff).is_ok());
        let dim3 = ir::dim::create(Arc::make_mut(fun), false);
        let new_objs = ir::NewObjs {
            dim: vec![dim3], basic_block: vec![dim3.into()], .. ir::NewObjs::default()
        };
        store.alloc(fun, &new_objs);
        store.set_bar(dim3, Bar::A);
        let actions = init_domain_partial(store, Arc::make_mut(fun), &new_objs, diff);
        assert!(apply_decisions(actions.unwrap(), fun, store).is_ok());
        assert!(!ir::dim::get(fun, dim2).condition());
        assert!(ir::dim::get(fun, dim3).condition());
    }
}

mod parametric_set {
    define_ir! { struct inst; struct operand[inst]; }
    generated_file!(parametric_set);
    use self::parametric_set::*;
    use std::sync::Arc;

    /// Ensures constraints on parametric sets are working.
    #[test]
    fn constraint() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let inst0 = ir::inst::create(&mut fun, false);
        let inst1 = ir::inst::create(&mut fun, false);
        let op0 = ir::operand::create(&mut fun, inst0, false);
        let op1 = ir::operand::create(&mut fun, inst1, false);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, &mut fun).unwrap();

        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());
        // Constrain the operand choice by setting the instruction choice.
        assert!(apply_decisions(vec![Action::Bar(inst0, Bar::D)], fun, store).is_ok());
        assert_eq!(store.get_foo(inst0, op0), Foo::A);
        // Constrain the instruction choice by setting the operand choice.
        assert!(apply_decisions(vec![Action::Foo(inst1, op1, Foo::B)], fun, store).is_ok());
        assert_eq!(store.get_bar(inst1), Bar::C);
        // Test post-lowering allocation and filtering.
        let op2 = ir::operand::create(Arc::make_mut(fun), inst0, false);
        let inst2 = ir::inst::create(Arc::make_mut(fun), false);
        let op3 = ir::operand::create(Arc::make_mut(fun), inst2, false);
        let new_ops = vec![(inst0, op2), (inst2, op3)];
        let new_objs = ir::NewObjs { inst: vec![inst2], operand: new_ops };
        store.alloc(fun, &new_objs);
        let actions = init_domain_partial(
            store, Arc::make_mut(fun), &new_objs, &mut DomainDiff::default()).unwrap();
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_foo(inst0, op2), Foo::A);
        assert_eq!(store.get_foo(inst2, op3), Foo::ALL);
    }
}

mod parametric_subset {
    define_ir! {
        // Define a set parameters, split in two.
        trait param;
        struct param_a: param;
        struct param_b: param;
        // Define two levels of parametric sets, all contained in a regular one.
        struct value;
        type value_ab[param reverse value]: value;
        type value_a[param_a reverse value]: value_ab[param];
        type value_b[param_b reverse value]: value_ab[param];
    }
    generated_file!(parametric_subset);
    use self::parametric_subset::*;
    use std::sync::Arc;

    /// Test set constraints involving parametric sets.
    #[test]
    fn set_constraints() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let param_a_0 = ir::param_a::create(&mut fun, false);
        let param_b_0 = ir::param_b::create(&mut fun, false);
        let param_a_1 = ir::param_a::create(&mut fun, false);
        let param_b_1 = ir::param_b::create(&mut fun, false);
        let value_a_0 = ir::value_a::create(&mut fun, param_a_0, false);
        let value_b_0 = ir::value_b::create(&mut fun, param_b_0, false);
        let value_a_1 = ir::value_a::create(&mut fun, param_a_1, false);
        let value_b_1 = ir::value_b::create(&mut fun, param_b_1, false);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());

        // Test a constraint on the parameter of a set.
        let actions = vec![
            Action::ValueAbChoice(param_a_0.into(), value_a_0.into(), ValueAbChoice::B),
            Action::ValueAbChoice(param_b_0.into(), value_b_0.into(), ValueAbChoice::B),
        ];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_param_choice(param_a_0.into()), ParamChoice::A);
        assert_eq!(store.get_param_choice(param_b_0.into()), ParamChoice::ALL);
        // Test a constraint on a parametric set, with no reversal.
        let (param_a_1, param_b_1) = (param_a_1.into(), param_b_1.into());
        let value_a_1: ir::value_ab::Id = value_a_1.into();
        let actions = vec![
            Action::ParamValueChoice(param_a_1, value_a_1.into(), ParamValueChoice::B),
        ];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_value_ab_choice(param_a_1, value_a_1), ValueAbChoice::A);
        assert_eq!(store.get_value_ab_choice(param_b_1, value_b_1.into()),
                   ValueAbChoice::ALL);
    }

    /// Test set constraints, where the set parameter is not yet defined.
    #[test]
    fn set_reversal() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let param_a = ir::param_a::create(&mut fun, false);
        let param_b = ir::param_b::create(&mut fun, false);
        let value_a = ir::value_a::create(&mut fun, param_a, false);
        let value_b = ir::value_b::create(&mut fun, param_b, false);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());

        // Test a reverse set constraint, with no additional constraint.
        let actions = vec![Action::ValueChoice(value_a.into(), ValueChoice::B)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_param_value_choice(param_a.into(), value_a.into()),
                   ParamValueChoice::A);
        assert_eq!(store.get_param_value_choice(param_b.into(), value_b.into()),
                   ParamValueChoice::ALL);
        // Test a reverse set constraint on an already constrained set.
        let actions = vec![Action::ValueChoice(value_b.into(), ValueChoice::B)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_param_value_choice_2(param_a.into(), value_a.into()),
                   ParamValueChoice2::A);
        assert_eq!(store.get_param_value_choice_2(param_b.into(), value_b.into()),
                   ParamValueChoice2::ALL);
    }
}

mod quotient_set {
    define_ir! {
        trait basic_block;
        struct inst: basic_block;
        struct dim: basic_block;
        type inst_quotient: inst;
        type dim_quotient[inst reverse dim]: dim;
    }
    generated_file!(quotient_set);
    use self::quotient_set::*;
    use std::sync::Arc;

    /// Callback that adds an instruction to the `InstQuotient` set.
    fn add_inst_to_quotient(fun: &mut ir::Function,
                            inst: ir::inst::Id) -> ir::NewObjs {
        if ir::inst_quotient::add_to_subset(fun, inst) {
            ir::NewObjs { inst_quotient: vec![inst], .. ir::NewObjs::default() }
        } else { ir::NewObjs::default() }
    }

    /// Callback that adds a dimension to the `DimQuotient(inst)` set.
    fn add_dim_to_quotient(fun: &mut ir::Function,
                           inst: ir::inst::Id,
                           dim: ir::dim::Id) -> ir::NewObjs {
        if ir::dim_quotient::add_to_subset(fun, inst, dim) {
            ir::NewObjs { dim_quotient: vec![(inst, dim)], .. ir::NewObjs::default() }
        } else { ir::NewObjs::default() }
    }

    /// Test quotient sets without arguments
    #[test]
    fn simple_set() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let mut maybe_repr: Vec<_> = (0..4)
            .map(|_| ir::inst::create(&mut fun, false)).collect();
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());

        // Ensure a single representative has been elected.
        let repr0 = {
            let mut iter = ir::inst_quotient::iter(fun);
            let repr = iter.next().unwrap().id();
            assert!(iter.next().is_none(), "only one instruction should be repr");
            repr
        };
        assert_eq!(store.get_repr_inst(repr0), Bool::TRUE);
        maybe_repr.retain(|&inst| inst != repr0);
        for &inst in &maybe_repr {
            assert_eq!(store.get_repr_inst(inst), Bool::ALL);
        }
        // Create a new class by setting the order to not MERGED
        let repr1 = maybe_repr.pop().unwrap();
        let actions = vec![Action::Order(repr1.into(), repr0.into(), !Order::MERGED)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_repr_inst(repr1), Bool::TRUE);
        for &inst in &maybe_repr {
            assert_eq!(store.get_repr_inst(inst), Bool::ALL);
        }
        // Create a repr by forcing the repr flag to TRUE
        let repr2 = maybe_repr.pop().unwrap();
        let actions = vec![Action::ReprInst(repr2.into(), Bool::TRUE)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_repr_inst(repr2), Bool::TRUE);
        assert_eq!(store.get_order(repr2.into(), repr0.into()), !Order::MERGED);
        assert_eq!(store.get_order(repr2.into(), repr1.into()), !Order::MERGED);
        for &inst in &maybe_repr {
            assert_eq!(store.get_repr_inst(inst), Bool::ALL);
        }
        // Set a repr flag to false and ensure the item is merged to anexisting repr.
        let not_repr = maybe_repr.pop().unwrap();
        assert!(maybe_repr.is_empty());
        let actions = vec![
            Action::Order(not_repr.into(), repr1.into(), !Order::MERGED),
            Action::Order(not_repr.into(), repr2.into(), !Order::MERGED),
            Action::ReprInst(not_repr, Bool::FALSE)
        ];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_order(not_repr.into(), repr0.into()), Order::MERGED);
    }

    /// Test quotient sets with an argument.
    #[test]
    fn parametric_set() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let inst0 = ir::inst::create(&mut fun, false);
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);
        let dim2 = ir::dim::create(&mut fun, false);
        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());
        // Ensure no representative are set after initialization.
        assert_eq!(store.get_active_dim(inst0, dim0), Bool::ALL);
        assert_eq!(store.get_active_dim(inst0, dim1), Bool::ALL);
        assert_eq!(store.get_active_dim(inst0, dim2), Bool::ALL);
        assert!(ir::dim_quotient::iter(fun, inst0).next().is_none());
        // Ensure a representative is selected after the dividend is set to true.
        let actions = vec![Action::Order(dim0.into(), inst0.into(), Order::OUTER)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_active_dim(inst0, dim0), Bool::TRUE);
        assert_eq!(ir::dim_quotient::iter(fun, inst0).next().map(|x| x.id()), Some(dim0));
        // Ensure the dividend is set to true if a representative is selected
        let actions = vec![Action::ActiveDim(inst0, dim1, Bool::TRUE)];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_order(dim1.into(), inst0.into()), Order::OUTER);
        assert_eq!(store.get_order(dim1.into(), dim0.into()), !Order::MERGED);
        assert_eq!(ir::dim_quotient::iter(fun, inst0).count(), 2);
        // Ensure the dividend is set to false if an item is set as not representative.
        let actions = vec![
            Action::ActiveDim(inst0, dim2, Bool::FALSE),
            Action::Order(dim2.into(), dim0.into(), !Order::MERGED),
            Action::Order(dim2.into(), dim1.into(), !Order::MERGED),
        ];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_order(dim2.into(), inst0.into()), !Order::OUTER);
    }

    /// Tests quotient sets initialization with a symmetric decision.
    fn symmetric_init() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let inst0 = ir::inst::create(&mut fun, false);
        let inst1 = ir::inst::create(&mut fun, false);
        let dim0 = ir::dim::create(&mut fun, false);
        let dim1 = ir::dim::create(&mut fun, false);

        let store = &mut DomainStore::new(&fun);
        store.set_order(dim0.into(), inst0.into(), Order::OUTER);
        store.set_order(dim0.into(), inst1.into(), Order::OUTER);
        store.set_order(dim1.into(), inst0.into(), Order::OUTER);
        store.set_order(dim1.into(), inst1.into(), Order::OUTER);
        store.set_order(dim0.into(), inst0.into(), Order::OUTER);
        store.set_order(dim0.into(), dim1.into(), Order::OUTER);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());

        assert_eq!(store.get_active_dim(inst0, dim0), Bool::TRUE);
        assert_eq!(store.get_active_dim(inst0, dim1), Bool::TRUE);
        assert_eq!(store.get_active_dim(inst1, dim0), Bool::TRUE);
        assert_eq!(store.get_active_dim(inst1, dim1), Bool::TRUE);
    }
}

mod integer_set {
    define_ir! { struct set0; }
    generated_file!(integer_set);
    use self::integer_set::*;
    use std::sync::Arc;

    const INT0_DOMAIN: [u32; 3] = [2, 4, 6];

    const INT1_DOMAIN: [u32; 3] = [3, 4, 5];

    fn int1_domain<'a>(_: &ir::Function, _: &'a ir::set0::Obj) -> &'a [u32] {
        &INT1_DOMAIN
    }

    #[test]
    fn domain() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let obj0 = ir::set0::create(&mut fun, false);
        let obj1 = ir::set0::create(&mut fun, false);

        let store = &mut DomainStore::new(&fun);
        store.set_int1(obj0, NumericSet::all(&[4]));
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());

        assert_eq!(store.get_int0(), NumericSet::all(&[2, 4]));
        assert_eq!(store.get_int1(obj1), NumericSet::all(&[4]));
    }

    #[test]
    fn counter() {
        let _ = ::env_logger::try_init();
        let mut fun = ir::Function::default();
        let obj0 = ir::set0::create(&mut fun, false);
        let obj1 = ir::set0::create(&mut fun, false);
        let obj2 = ir::set0::create(&mut fun, false);

        let store = &mut DomainStore::new(&fun);
        let actions = init_domain(store, &mut fun).unwrap();
        let fun = &mut Arc::new(fun);
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_sum_int2(), Range { min: 9, max: 15 });

        let actions = vec![Action::Int2(obj0, NumericSet::all(&[4]))];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_sum_int2(), Range { min: 10, max: 14 });

        let actions = vec![Action::Int2(obj1, NumericSet::all(&[5]))];
        assert!(apply_decisions(actions, fun, store).is_ok());
        assert_eq!(store.get_sum_int2(), Range { min: 12, max: 12 });
        assert_eq!(store.get_int2(obj2), NumericSet::all(&[3]));

    }
}
