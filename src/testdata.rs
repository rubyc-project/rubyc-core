//! Shared sample programs used by tests, the CLI, and any external
//! consumer that needs known-good RubyC sources.

/// A named source snippet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSample {
    pub name: &'static str,
    pub source: &'static str,
}

/// Block-form namespace, Allman braces.
pub const BLOCK_FORM: &str = r#"
namespace testnc
{
    class testc
    {
        int main()
        {
            printf("Hello World!");
        }
    }
}
"#;

/// File-scoped namespace form.
pub const FILE_SCOPED_FORM: &str = r#"
namespace testnc;
class testc
{
    int main()
    {
        printf("Hello World!");
    }
}
"#;

/// Brace on the previous line (K&R / Rust style).
pub const KN_R_STYLE: &str = "
namespace testnc {
    class testc {
        int main() {
            printf(\"Hello World!\");
        }
    }
}
";

/// The whole program compressed onto a single line.
pub const ONE_LINE: &str =
    "namespace testnc { class testc { int main() { printf(\"Hello World!\"); } } }";

/// All hello-world samples. Every one must parse to the same structure.
pub const SAMPLES: &[SourceSample] = &[
    SourceSample {
        name: "block_form",
        source: BLOCK_FORM,
    },
    SourceSample {
        name: "file_scoped_form",
        source: FILE_SCOPED_FORM,
    },
    SourceSample {
        name: "kn_r_style",
        source: KN_R_STYLE,
    },
    SourceSample {
        name: "one_line",
        source: ONE_LINE,
    },
];
