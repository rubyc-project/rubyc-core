//! Lint rules (W0001 unused-local, W0002 unused-private-member) and the
//! `LintConfig` / `Level` machinery: resolution, overrides, and how the
//! severity of a diagnostic tracks the configured level.

use crate::lints::{analyze, classes_of, LintConfig, Level, RULES, LintRule};
use crate::parser;

fn parse(src: &str) -> crate::models::CompilationUnit {
    parser::parse(src)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}\nsrc:\n{src}"))
}

// ---------- rule registry ----------

#[test]
fn registry_has_stable_ids() {
    assert_eq!(RULES.len(), 2);
    assert_eq!(RULES[0].id, "W0001");
    assert_eq!(RULES[0].name, "unused-local");
    assert_eq!(RULES[1].id, "W0002");
    assert_eq!(RULES[1].name, "unused-private-member");
    // Both default to Warn.
    assert_eq!(RULES[0].default, Level::Warn);
    assert_eq!(RULES[1].default, Level::Warn);
}

#[test]
fn by_key_resolves_id_and_name() {
    assert_eq!(LintRule::by_key("W0001").map(|r| r.name), Some("unused-local"));
    assert_eq!(LintRule::by_key("unused-local").map(|r| r.id), Some("W0001"));
    assert_eq!(LintRule::by_key("W0002").map(|r| r.name), Some("unused-private-member"));
    assert!(LintRule::by_key("nope").is_none());
}

// ---------- LintConfig resolution ----------

#[test]
fn config_defaults_to_warn_when_unset() {
    let cfg = LintConfig::default();
    assert_eq!(cfg.level_for("W0001"), Level::Warn);
    assert_eq!(cfg.level_for("W0002"), Level::Warn);
}

#[test]
fn config_unknown_rule_is_allow() {
    let cfg = LintConfig::default();
    assert_eq!(cfg.level_for("W9999"), Level::Allow);
}

#[test]
fn config_override_by_id() {
    let mut cfg = LintConfig::default();
    cfg.set("W0001", Level::Deny);
    assert_eq!(cfg.level_for("W0001"), Level::Deny);
    // Other rule unaffected.
    assert_eq!(cfg.level_for("W0002"), Level::Warn);
}

#[test]
fn config_override_by_name() {
    let mut cfg = LintConfig::default();
    cfg.set("unused-local", Level::Allow);
    assert_eq!(cfg.level_for("W0001"), Level::Allow);
}

#[test]
fn config_later_override_wins() {
    let mut cfg = LintConfig::default();
    cfg.set("W0001", Level::Deny);
    cfg.set("W0001", Level::Allow);
    assert_eq!(cfg.level_for("W0001"), Level::Allow);
}

// ---------- W0001: unused local ----------

#[test]
fn unused_local_warns_by_default() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = 5;
                    return 1;
                }
            }
        }",
    );
    let diags = analyze(&unit, &LintConfig::default());
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code, "W0001");
    assert_eq!(diags[0].severity, crate::models::Severity::Warning);
}

#[test]
fn used_local_not_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = 5;
                    return x;
                }
            }
        }",
    );
    assert!(analyze(&unit, &LintConfig::default()).is_empty());
}

#[test]
fn pure_write_does_not_count_as_read() {
    // `x` is assigned twice but never read -> still dead.
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = 5;
                    x = 10;
                    return 1;
                }
            }
        }",
    );
    assert_eq!(analyze(&unit, &LintConfig::default()).len(), 1);
}

#[test]
fn deny_promotes_to_error() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = 5;
                    return 1;
                }
            }
        }",
    );
    let mut cfg = LintConfig::default();
    cfg.set("W0001", Level::Deny);
    cfg.set("W0002", Level::Allow);
    let diags = analyze(&unit, &cfg);
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].severity, crate::models::Severity::Error);
}

#[test]
fn allow_suppresses() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = 5;
                    return 1;
                }
            }
        }",
    );
    let mut cfg = LintConfig::default();
    cfg.set("W0001", Level::Allow);
    assert!(analyze(&unit, &cfg).is_empty());
}

#[test]
fn read_inside_nested_control_counts() {
    // `x` read inside an if -> live.
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int x = 5;
                    if (x > 3) {
                        return 1;
                    }
                    return 0;
                }
            }
        }",
    );
    assert!(analyze(&unit, &LintConfig::default()).is_empty());
}

#[test]
fn two_unused_locals_both_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                public int f() {
                    int a = 1;
                    int b = 2;
                    return 1;
                }
            }
        }",
    );
    assert_eq!(analyze(&unit, &LintConfig::default()).len(), 2);
}

// ---------- W0002: unused private member ----------

#[test]
fn unused_private_method_warns() {
    let unit = parse(
        "namespace n {
            class c {
                int main() {
                    return 0;
                }
                private int helper() {
                    return 0;
                }
            }
        }",
    );
    let diags = analyze(&unit, &LintConfig::default());
    // Only the private `helper` is flagged; `main` is a root.
    assert_eq!(diags.len(), 1);
    assert_eq!(diags[0].code, "W0002");
}

#[test]
fn referenced_private_method_not_flagged() {
    let unit = parse(
        "namespace n {
            class c {
                int main() {
                    return helper();
                }
                private int helper() {
                    return 0;
                }
            }
        }",
    );
    assert!(analyze(&unit, &LintConfig::default()).is_empty());
}

#[test]
fn main_is_never_flagged() {
    // A `main` that references nothing is a root, never dead.
    let unit = parse(
        "namespace n {
            class c {
                int main() {
                    return 0;
                }
            }
        }",
    );
    assert!(analyze(&unit, &LintConfig::default()).is_empty());
}

#[test]
fn public_method_not_flagged_even_if_unused() {
    let unit = parse(
        "namespace n {
            class c {
                int main() {
                    return 0;
                }
                public int helper() {
                    return 0;
                }
            }
        }",
    );
    // W0002 only targets private/default members.
    assert!(analyze(&unit, &LintConfig::default()).is_empty());
}

#[test]
fn unused_private_field_warns() {
    let unit = parse(
        "namespace n {
            class c {
                private int stored = 0;
                int main() {
                    return 0;
                }
            }
        }",
    );
    let diags = analyze(&unit, &LintConfig::default());
    assert!(diags.iter().any(|d| d.code == "W0002"));
}

// ---------- classes_of ----------

#[test]
fn classes_of_returns_class_decls() {
    let unit = parse(
        "namespace n {
            class a { int main() { return 0; } }
            class b { int f() { return 0; } }
        }",
    );
    let names: Vec<String> = classes_of(&unit).iter().map(|c| c.name.clone()).collect();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn classes_of_empty_when_no_classes() {
    // A unit with no class members.
    let unit = parse("");
    assert!(classes_of(&unit).is_empty());
}
