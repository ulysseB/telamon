//! Tests that failed at some point.

mod fail0 {
    define_ir! {
        trait set_a;
        struct subset_a[subset_b reverse set_a]: set_a;

        trait set_b;
        struct subset_b: set_b;
    }

    generated_file!(fail0);
}
