//! Central CLI specification tables.
//!
//! Every flag and command the tool accepts lives here — these tables drive
//! both argument parsing and help rendering, so documentation cannot drift
//! from behavior.

use super::{Command, IoArgs, OutFormat};

#[derive(Clone, Copy, PartialEq)]
pub enum ArgKind {
    /// Boolean flag, takes nothing.
    Switch,
    /// Requires a value: `--out x` or `--out=x`.
    Required,
    /// Value optional: `--shebang` uses the default, `--shebang=/path` overrides.
    OptionalValue { default: &'static str },
}

#[derive(Clone, Copy)]
pub struct FlagSpec {
    pub short: Option<char>,
    pub long: &'static str,
    pub kind: ArgKind,
    /// Allowed values for enum-valued flags; empty means free-form text.
    pub choices: &'static [&'static str],
    /// Placeholder shown in help (`FILE`, `FMT`, ...).
    pub placeholder: &'static str,
    pub help: &'static str,
}

impl FlagSpec {
    const fn switch(short: char, long: &'static str, help: &'static str) -> Self {
        Self {
            short: Some(short),
            long,
            kind: ArgKind::Switch,
            choices: &[],
            placeholder: "",
            help,
        }
    }

    const fn value(
        short: char,
        long: &'static str,
        placeholder: &'static str,
        help: &'static str,
    ) -> Self {
        Self {
            short: Some(short),
            long,
            kind: ArgKind::Required,
            choices: &[],
            placeholder,
            help,
        }
    }

    pub(super) fn matches_long(&self, long: &str) -> bool {
        self.long == long
    }

    pub(super) fn matches_short(&self, c: char) -> bool {
        self.short == Some(c)
    }
}

pub struct CommandSpec {
    pub name: &'static str,
    pub about: &'static str,
    pub flags: &'static [FlagSpec],
    pub examples: &'static str,
}

pub(super) const IN_FORMAT_FLAG: FlagSpec = FlagSpec {
    short: Some('I'),
    long: "in-format",
    kind: ArgKind::Required,
    choices: &["auto", "source", "bytecode"],
    placeholder: "FMT",
    help: "rubyc.cli.flag.in_format",
};

pub(super) const OUT_FORMAT_FLAG: FlagSpec = FlagSpec {
    short: Some('O'),
    long: "out-format",
    kind: ArgKind::Required,
    choices: &[
        "bytecode",
        "bytecode-shebang",
        "base64-bytecode",
        "exe",
        "shared",
        "static",
    ],
    placeholder: "FMT",
    help: "rubyc.cli.flag.out_format",
};

pub const GLOBAL_FLAGS: &[FlagSpec] = &[
    FlagSpec::value('D', "define", "SYM", "rubyc.cli.flag.define"),
    FlagSpec::switch('v', "verbose", "rubyc.cli.flag.verbose"),
    FlagSpec::switch('q', "quiet", "rubyc.cli.flag.quiet"),
];

const IO_FLAGS: &[FlagSpec] = &[
    FlagSpec::value('i', "in", "FILE", "rubyc.cli.flag.in"),
    FlagSpec::value('o', "out", "FILE", "rubyc.cli.flag.out"),
    IN_FORMAT_FLAG,
    OUT_FORMAT_FLAG,
    FlagSpec::switch('B', "base64", "rubyc.cli.flag.base64"),
    FlagSpec::switch('b', "bytecode", "rubyc.cli.flag.bytecode"),
    FlagSpec {
        short: None,
        long: "shebang",
        kind: ArgKind::OptionalValue {
            default: "/usr/bin/env rubyc",
        },
        choices: &[],
        placeholder: "INTERPRETER",
        help: "rubyc.cli.flag.shebang",
    },
    FlagSpec::switch('g', "debug", "rubyc.cli.flag.debug"),
    FlagSpec::switch('m', "no-magic", "rubyc.cli.flag.no_magic"),
    FlagSpec::switch('P', "skip-pre-hook", "rubyc.cli.flag.skip_pre_hook"),
    FlagSpec::switch('Q', "skip-post-hook", "rubyc.cli.flag.skip_post_hook"),
    FlagSpec {
        short: None,
        long: "rci-only",
        kind: ArgKind::Switch,
        choices: &[],
        placeholder: "",
        help: "rubyc.cli.flag.rci_only",
    },
    FlagSpec::value('j', "jobs", "N", "rubyc.cli.flag.jobs"),
    // Last: every command selects flags by numeric index, so appended
    // flags must never shift existing positions.
    FlagSpec::value('l', "lib", "FILE", "rubyc.cli.flag.lib"),
];

/// `new --solution`: scaffold a `.rcs` solution instead of a `.rcp` project.
pub(super) const SOLUTION_FLAG: FlagSpec =
    FlagSpec::switch('s', "solution", "rubyc.cli.flag.solution");

pub const NEW: CommandSpec = CommandSpec {
    name: "new",
    about: "rubyc.cli.NEW.about",
    flags: &[SOLUTION_FLAG],
    examples: "rubyc.cli.NEW.examples",
};

pub const CREATE: CommandSpec = CommandSpec {
    name: "create",
    about: "rubyc.cli.CREATE.about",
    flags: &[SOLUTION_FLAG],
    examples: "rubyc.cli.CREATE.examples",
};

pub const ADD: CommandSpec = CommandSpec {
    name: "add",
    about: "rubyc.cli.ADD.about",
    flags: &[],
    examples: "rubyc.cli.ADD.examples",
};

pub const COMPILE: CommandSpec = CommandSpec {
    name: "compile",
    about: "rubyc.cli.COMPILE.about",
    flags: IO_FLAGS,
    examples: "rubyc.cli.COMPILE.examples",
};

pub const BUILD: CommandSpec = CommandSpec {
    name: "build",
    about: "rubyc.cli.BUILD.about",
    flags: &[
        IO_FLAGS[0],
        IO_FLAGS[1],
        IO_FLAGS[3],
        IO_FLAGS[4],
        IO_FLAGS[5],
        IO_FLAGS[6],
        IO_FLAGS[7],
        IO_FLAGS[8],
        IO_FLAGS[9],
        IO_FLAGS[10],
        IO_FLAGS[11],
        IO_FLAGS[12],
        IO_FLAGS[13],
    ],
    examples: "rubyc.cli.BUILD.examples",
};

pub const RUN: CommandSpec = CommandSpec {
    name: "run",
    about: "rubyc.cli.RUN.about",
    flags: &[IO_FLAGS[0], IO_FLAGS[4], IO_FLAGS[5]],
    examples: "rubyc.cli.RUN.examples",
};

pub const DUMP: CommandSpec = CommandSpec {
    name: "dump",
    about: "rubyc.cli.DUMP.about",
    flags: &[IO_FLAGS[0], IO_FLAGS[4]],
    examples: "rubyc.cli.DUMP.examples",
};

pub const CHECK_ROUNDTRIP: CommandSpec = CommandSpec {
    name: "check-roundtrip",
    about: "rubyc.cli.CHECK_ROUNDTRIP.about",
    flags: &[],
    examples: "rubyc.cli.CHECK_ROUNDTRIP.examples",
};

pub const LINT: CommandSpec = CommandSpec {
    name: "lint",
    about: "rubyc.cli.LINT.about",
    flags: &[IO_FLAGS[0], IO_FLAGS[4]],
    examples: "rubyc.cli.LINT.examples",
};

pub const LIST: CommandSpec = CommandSpec {
    name: "list",
    about: "rubyc.cli.LIST.about",
    flags: &[
        FlagSpec {
            short: None,
            long: "kind",
            kind: ArgKind::Required,
            choices: &["all", "target"],
            placeholder: "KIND",
            help: "rubyc.cli.flag.kind",
        },
        FlagSpec {
            short: None,
            long: "remote",
            kind: ArgKind::Switch,
            choices: &[],
            placeholder: "",
            help: "rubyc.cli.flag.remote",
        },
    ],
    examples: "rubyc.cli.LIST.examples",
};

pub const INSTALL: CommandSpec = CommandSpec {
    name: "install",
    about: "rubyc.cli.INSTALL.about",
    flags: &[],
    examples: "rubyc.cli.INSTALL.examples",
};

pub const DOWNLOAD: CommandSpec = CommandSpec {
    name: "download",
    about: "rubyc.cli.DOWNLOAD.about",
    flags: &[
        FlagSpec {
            short: None,
            long: "url",
            kind: ArgKind::Required,
            choices: &[],
            placeholder: "URL",
            help: "rubyc.cli.flag.url",
        },
        FlagSpec::value('t', "type", "KIND", "rubyc.cli.flag.type"),
    ],
    examples: "rubyc.cli.DOWNLOAD.examples",
};

pub const REMOVE: CommandSpec = CommandSpec {
    name: "remove",
    about: "rubyc.cli.REMOVE.about",
    flags: &[
        FlagSpec {
            short: None,
            long: "kind",
            kind: ArgKind::Required,
            choices: &["target"],
            placeholder: "KIND",
            help: "rubyc.cli.flag.kind",
        },
    ],
    examples: "rubyc.cli.REMOVE.examples",
};

pub const UPDATE: CommandSpec = CommandSpec {
    name: "update",
    about: "rubyc.cli.UPDATE.about",
    flags: &[],
    examples: "rubyc.cli.UPDATE.examples",
};

/// Target discovery and selection flags, accepted by every command.
///
/// `--base` points at an installation root (its `targets/` subdirectory is
/// scanned); the per-kind flags override it; `--target` selects one discovered
/// target. Languages are in-process, not plugins, so there is no language
/// selection.
pub const EXTENSION_FLAGS: &[FlagSpec] = &[
    FlagSpec {
        short: None,
        long: "base",
        kind: ArgKind::Required,
        choices: &[],
        placeholder: "DIR",
        help: "rubyc.cli.flag.base",
    },
    FlagSpec {
        short: None,
        long: "targets-dir",
        kind: ArgKind::Required,
        choices: &[],
        placeholder: "DIR",
        help: "rubyc.cli.flag.targets_dir",
    },
    FlagSpec::value('t', "target", "NAME", "rubyc.cli.flag.target"),
];

pub const COMMANDS: &[CommandSpec] = &[
    COMPILE,
    BUILD,
    RUN,
    LINT,
    DUMP,
    LIST,
    INSTALL,
    DOWNLOAD,
    REMOVE,
    UPDATE,
    CHECK_ROUNDTRIP,
    NEW,
    CREATE,
    ADD,
];

// ---- lookup / application helpers shared by the parser ----

pub(super) fn is_known_head(word: &str) -> bool {
    word == "help" || COMMANDS.iter().any(|c| c.name == word)
}

pub(super) fn find_command(name: &str) -> Option<&'static CommandSpec> {
    COMMANDS.iter().find(|c| c.name == name)
}

pub(super) fn command_from_name(name: &str) -> Command {
    match name {
        "compile" => Command::Compile,
        "build" => Command::Build,
        "run" => Command::Run,
        "lint" => Command::Lint,
        "dump" => Command::Dump,
        "check-roundtrip" => Command::CheckRoundtrip,
        "list" => Command::List,
        "list-remote" => Command::ListRemote,
        "install" => Command::Install,
        "download" => Command::Download,
        "remove" => Command::Remove,
        "update" => Command::Update,
        "new" => Command::New,
        "create" => Command::Create,
        "add" => Command::Add,
        other => unreachable!("unknown command passed guard: {other}"),
    }
}

pub(super) fn spec_for(command: Option<Command>) -> Option<&'static CommandSpec> {
    command.map(|c| match c {
        Command::Compile => &COMPILE,
        Command::Build => &BUILD,
        Command::Run => &RUN,
        Command::Lint => &LINT,
        Command::Dump => &DUMP,
        Command::CheckRoundtrip => &CHECK_ROUNDTRIP,
        Command::List => &LIST,
        Command::Install => &INSTALL,
        Command::Download => &DOWNLOAD,
        Command::Remove => &REMOVE,
        Command::Update => &UPDATE,
        Command::ListRemote => &LIST,
        Command::New => &NEW,
        Command::Create => &CREATE,
        Command::Add => &ADD,
    })
}

pub(super) fn apply_switch(flag: &FlagSpec, verbose: &mut bool, quiet: &mut bool, io: &mut IoArgs) {
    match flag.long {
        "verbose" => *verbose = true,
        "quiet" => *quiet = true,
        "base64" => io.base64 = true,
        "debug" => io.debug = true,
        "bytecode" => {
            io.out_format = Some(OutFormat::Bytecode);
        }
        "solution" => io.solution = true,
        "no-magic" => io.no_magic = true,
        "skip-pre-hook" => io.skip_pre_hook = true,
        "skip-post-hook" => io.skip_post_hook = true,
        "rci-only" => io.rci_only = true,
        "remote" => {}
        _ => unreachable!("unhandled switch flag '{}'", flag.long),
    }
}

pub(super) fn default_of(flag: &FlagSpec) -> &'static str {
    match flag.kind {
        ArgKind::OptionalValue { default } => default,
        _ => "",
    }
}

pub(super) fn store_value(
    flag: &FlagSpec,
    value: &str,
    io: &mut IoArgs,
    defines: &mut Vec<String>,
) -> std::result::Result<(), crate::cli::CliError> {
    use crate::cli::{InFormat, OutFormat};
    match flag.long {
        "define" => defines.push(value.to_owned()),
        "in" => io.input = Some(value.to_owned()),
        "in-format" => {
            io.in_format = Some(InFormat::from_str(value).ok_or_else(|| {
                crate::cli::CliError::InvalidValue {
                    flag: "--in".into(),
                    value: value.to_owned(),
                    allowed: vec!["auto".into(), "source".into(), "bytecode".into()],
                }
            })?)
        }
        "out-format" => {
            io.out_format = Some(OutFormat::from_str(value).ok_or_else(|| {
                crate::cli::CliError::InvalidValue {
                    flag: "-O".into(),
                    value: value.to_owned(),
                    allowed: [
                        "bytecode",
                        "bytecode-shebang",
                        "base64-bytecode",
                        "exe",
                        "shared",
                        "static",
                    ]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                }
            })?)
        }
        "out" => io.output = Some(value.to_owned()),
        "file" => io.file = Some(value.to_owned()),
        "input" => io.input = Some(value.to_owned()),
        "base64" => io.base64 = true,
        "shebang" => io.shebang = Some(value.to_owned()),
        "lib" => io.libs.push(value.to_owned()),
        "jobs" => {
            io.jobs = value.parse::<usize>().map_err(|_| crate::cli::CliError::InvalidValue {
                flag: "--jobs".into(),
                value: value.to_owned(),
                allowed: vec!["a positive integer".into()],
            })?
        }
        "base" => io.base = Some(value.to_owned()),
        "targets-dir" => io.targets_dir = Some(value.to_owned()),
        "target" => io.target = Some(value.to_owned()),
        "kind" => io.kind = Some(value.to_owned()),
        "url" => io.url = Some(value.to_owned()),
        "type" => {}
        other => unreachable!("unhandled value flag '{other}'"),
    }
    Ok(())
}
