//! Tests that failed at some point.

mod fail0 {
    define_ir! {
        struct set_a;
        type subset_a[subset_b reverse set_a]: set_a;

        trait set_b;
        struct subset_b: set_b;
    }

    generated_file!(fail0);
}
