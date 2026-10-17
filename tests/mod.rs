//! Unit tests for rubyc-core.
//!
//! Every directory mirrors a compiler module and each file names the behavior
//! it covers. Tests are compiled inside the crate so they can
//! reach crate-private internals without inflating the public API;
//! anything a suite needs beyond its own module is `pub(crate)` at most.
//!
//! End-to-end suites that spawn the `rubyc` binary live in the same-named
//! `tests/rubyc-cli/` tree.

// parser/
#[path = "parser/lexer_tokens.rs"] mod parser_lexer_tokens;
#[path = "parser/unit_shapes.rs"] mod parser_unit_shapes;
#[path = "parser/error_recovery.rs"] mod parser_error_recovery;
#[path = "parser/assembly_encoding.rs"] mod parser_assembly_encoding;
#[path = "parser/assembly_full.rs"] mod parser_assembly_full;
#[path = "parser/comments.rs"] mod comments_handling;

// bytecode/
#[path = "bytecode/roundtrip_identity.rs"] mod bytecode_roundtrip_identity;
#[path = "bytecode/no_magic.rs"] mod bytecode_no_magic;

// models/
#[path = "models/traits_derives.rs"] mod models_traits_derives;
#[path = "models/visitor_traversal.rs"] mod models_visitor_traversal;
#[path = "models/coverage_net.rs"] mod models_coverage_net;

// ir + native semantic
#[path = "native/ir_constant_folding.rs"] mod ir_constant_folding;
#[path = "native/ir_container_roundtrip.rs"] mod ir_container_roundtrip;
#[path = "native/rci_roundtrip.rs"] mod native_rci_roundtrip;
#[path = "native/async_smoke.rs"] mod native_async_smoke;
#[path = "native/ir_chained_field.rs"] mod native_ir_chained_field;
#[path = "native/virtual_property_dispatch.rs"] mod native_virtual_property_dispatch;
#[path = "native/float_lowering.rs"] mod native_float_lowering;
#[path = "native/postfix_incdec.rs"] mod native_postfix_incdec;
#[path = "native/string_lowering.rs"] mod native_string_lowering;
#[path = "native/optional_args.rs"] mod native_optional_args;
#[path = "native/switch_scoping.rs"] mod native_switch_scoping;
#[path = "native/extension_calls.rs"] mod native_extension_calls;
#[path = "native/dep_import_probe.rs"] mod native_dep_import_probe;
#[path = "native/exe_dep_guard.rs"] mod native_exe_dep_guard;
#[path = "native/semantic_type_table.rs"] mod native_semantic_type_table;
#[path = "native/nested_resolution.rs"] mod native_nested_resolution;
#[path = "native/out_vars.rs"] mod native_out_vars;
#[path = "native/overload_access.rs"] mod native_overload_access;
#[path = "native/import_marshal.rs"] mod native_import_marshal;

// preprocessor/
#[path = "preprocessor/directives.rs"] mod preprocessor_directives;

// lints/
#[path = "lints/rules.rs"] mod lints_rules;

// inliner/
#[path = "inliner/expansion.rs"] mod inliner_expansion;

// cli/
#[path = "cli/args_parsing.rs"] mod cli_args_parsing;

// driver/
#[path = "driver/orchestration.rs"] mod driver_orchestration;
#[path = "driver/closure_defines.rs"] mod driver_closure_defines;

// project/
#[path = "project/source_glob.rs"] mod project_source_glob;

// plugin/
#[path = "plugin/loader.rs"] mod plugin_loader;

// diagnostics, i18n, text utilities
#[path = "diagnostics/rendering.rs"] mod diagnostics_rendering;
#[path = "diagnostics/semantic_scan.rs"] mod diagnostics_semantic_scan;
#[path = "diagnostics/i18n_lookup_and_fallback.rs"] mod i18n_lookup_and_fallback;
#[path = "utilities/format_interpolate.rs"] mod utilities_format_interpolate;
#[path = "utilities/base64_codec.rs"] mod utilities_base64_codec;
#[path = "utilities/suggest_did_you_mean.rs"] mod utilities_suggest_did_you_mean;
#[path = "extensions/discovery.rs"] mod extensions_discovery;




#[path = "native/zz_argv.rs"] mod zz_argv;
