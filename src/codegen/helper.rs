use std::fmt;

/// A struct to indent on new lines when writing formatting traits.
///
/// This is inspired from the [`PadAdapter`] from the standard library, but without using private
/// [`std::fmt::Formatter`] APIs.
///
/// [`PadAdapter`]: https://github.com/rust-lang/rust/blob/316a391dcb7d66dc25f1f9a4ec9d368ef7615005/src/libcore/fmt/builders.rs
pub struct IndentAdapter<'a> {
    fmt: &'a mut (dyn fmt::Write + 'a),
    on_newline: bool,
    padding: &'a str,
}

impl<'a> IndentAdapter<'a> {
    /// Create a new [`IndentAdapter`].std
    ///
    /// # Notes
    ///
    /// This assumes that the adapter is created after a newline; indentation will be added to the
    /// first formatted value.
    pub fn new<'b: 'a>(fmt: &'a mut fmt::Formatter<'b>) -> Self {
        IndentAdapter::with_prefix(fmt, "  ")
    }

    pub fn with_prefix<'b: 'a>(
        fmt: &'a mut fmt::Formatter<'b>,
        padding: &'a str,
    ) -> Self {
        IndentAdapter {
            fmt,
            on_newline: true,
            padding,
        }
    }
}

impl fmt::Write for IndentAdapter<'_> {
    fn write_str(&mut self, mut s: &str) -> fmt::Result {
        while !s.is_empty() {
            if self.on_newline {
                self.fmt.write_str(self.padding)?;
            }

            let split = match s.find('\n') {
                Some(pos) => {
                    self.on_newline = true;
                    pos + 1
                }
                None => {
                    self.on_newline = false;
                    s.len()
                }
            };

            self.fmt.write_str(&s[..split])?;
            s = &s[split..];
        }

        Ok(())
    }
}
