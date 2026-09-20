//! Hand-rolled argv parser. Exits are never performed here; the caller
//! decides what to do with [`super::ParsedArgs::Help`] / `Version`.

use super::help::{render_help, version_text};
use super::spec::{
    ArgKind, EXTENSION_FLAGS, GLOBAL_FLAGS, apply_switch, command_from_name, default_of,
    find_command, is_known_head, spec_for, store_value,
};
use super::{CliError, Command, ErrorFormat, Invocation, IoArgs, ParsedArgs};

/// How many positional arguments this command accepts.
fn positionals_for(command: Command) -> usize {
    match command {
        Command::List => 0,
        Command::Build
        | Command::Run
        | Command::Lint
        | Command::Compile
        | Command::Dump
        | Command::CheckRoundtrip
        | Command::Install
        | Command::Remove
        | Command::Update
        | Command::ListRemote
        | Command::Download
        | Command::New
        | Command::Create
        | Command::Add => 1,
    }
}

/// Whether a command's positional is a file path (vs a package name).
fn positional_is_file(command: Command) -> bool {
    matches!(
        command,
        Command::Build
            | Command::Run
            | Command::Lint
            | Command::Compile
            | Command::Dump
            | Command::CheckRoundtrip
            | Command::Add
    )
}

pub fn parse_args(argv: &[String]) -> Result<ParsedArgs, CliError> {
    // Shebang scripts invoke `rubyc ./script` with no subcommand.
    let mut args: Vec<String> = argv.to_vec();
    if args.len() > 1 && !is_known_head(&args[1]) && !args[1].starts_with('-') {
        args.insert(1, "run".to_string());
    }

    let mut verbose = false;
    let mut quiet = false;
    let mut no_color = false;
    let mut error_format = ErrorFormat::Text;
    let mut io = IoArgs::default();
    let mut command: Option<Command> = None;
    let mut positionals_left = 1usize;
    let mut end_of_options = false;
    let mut lint_overrides: Vec<(String, crate::lints::Level)> = Vec::new();
    let mut defines: Vec<String> = Vec::new();

    let mut i = 1usize;
    while i < args.len() {
        let arg = args[i].clone();

        if !end_of_options && arg == "--" {
            end_of_options = true;
            i += 1;
            continue;
        }

        if !end_of_options && arg.starts_with("--") {
            let body = &arg[2..];
            let (name, inline_value) = match body.split_once('=') {
                Some((n, v)) => (n, Some(v.to_owned())),
                None => (body, None),
            };
            match name {
                "help" => return Ok(ParsedArgs::Help(render_help(command))),
                "version" => return Ok(ParsedArgs::Version(version_text())),
                "no-color" => {
                    no_color = true;
                    i += 1;
                    continue;
                }
                "define" => {
                    let value = match inline_value {
                        Some(v) => v,
                        None => match args.get(i + 1) {
                            Some(v) if !v.starts_with('-') => {
                                i += 1;
                                v.clone()
                            }
                            _ => return Err(CliError::MissingValue(arg.clone())),
                        },
                    };
                    defines.push(value);
                    i += 1;
                    continue;
                }
                "warn" | "deny" | "allow" => {
                    let level = match name {
                        "warn" => crate::lints::Level::Warn,
                        "deny" => crate::lints::Level::Deny,
                        _ => crate::lints::Level::Allow,
                    };
                    let value = match inline_value {
                        Some(v) => v,
                        None => match args.get(i + 1) {
                            Some(v) if !v.starts_with('-') => {
                                i += 1;
                                v.clone()
                            }
                            _ => return Err(CliError::MissingValue(arg.clone())),
                        },
                    };
                    if crate::lints::LintRule::by_key(&value).is_none() {
                        return Err(CliError::InvalidValue {
                            flag: format!("--{name}"),
                            value,
                            allowed: crate::lints::RULES
                                .iter()
                                .flat_map(|r| [r.id.to_string(), r.name.to_string()])
                                .collect(),
                        });
                    }
                    lint_overrides.push((value, level));
                    i += 1;
                    continue;
                }
                "lib" => {
                    let value = match inline_value {
                        Some(v) => v,
                        None => match args.get(i + 1) {
                            Some(v) if !v.starts_with('-') => {
                                i += 1;
                                v.clone()
                            }
                            _ => return Err(CliError::MissingValue(arg.clone())),
                        },
                    };
                    io.libs.push(value);
                    i += 1;
                    continue;
                }
                "error-format" => {
                    let value = match inline_value {
                        Some(v) => v,
                        None => match args.get(i + 1) {
                            Some(v) if !v.starts_with('-') => {
                                i += 1;
                                v.clone()
                            }
                            _ => return Err(CliError::MissingValue(arg.clone())),
                        },
                    };
                    error_format = match value.as_str() {
                        "text" => ErrorFormat::Text,
                        "json" => ErrorFormat::Json,
                        "json-pretty" => ErrorFormat::JsonPretty,
                        other => {
                            return Err(CliError::InvalidValue {
                                flag: "--error-format".into(),
                                value: other.into(),
                                allowed: vec!["text".into(), "json".into(), "json-pretty".into()],
                            });
                        }
                    };
                    i += 1;
                    continue;
                }
                _ => {}
            }
            let flag = EXTENSION_FLAGS
                .iter()
                .chain(GLOBAL_FLAGS.iter())
                .chain(
                    spec_for(command)
                        .map(|s| s.flags.iter())
                        .unwrap_or([].iter()),
                )
                .find(|f| f.matches_long(name));
            let flag = flag.ok_or_else(|| CliError::UnknownFlag(arg.clone()))?;
            let mut consumed_extra = false;
            let resolved = match (flag.kind, inline_value) {
                (_, Some(v)) => Some(v),
                (ArgKind::Switch, None) => None,
                (kind @ ArgKind::OptionalValue { .. }, None) => {
                    // Only use the default when the next token is another flag;
                    // otherwise an attached-looking value still wins via '='.
                    _ = kind;
                    Some(default_of(flag).to_owned())
                }
                (ArgKind::Required, None) => {
                    let next = args
                        .get(i + 1)
                        .ok_or_else(|| CliError::MissingValue(arg.clone()))?;
                    if next.starts_with('-') {
                        return Err(CliError::MissingValue(arg.clone()));
                    }
                    consumed_extra = true;
                    Some(next.clone())
                }
            };
            if let Some(value) = resolved {
                store_value(flag, &value, &mut io, &mut defines)?;
            }
            if matches!(flag.kind, ArgKind::Switch) {
                apply_switch(flag, &mut verbose, &mut quiet, &mut io);
            }
            i += if consumed_extra { 2 } else { 1 };
            continue;
        }

        if !end_of_options && arg.starts_with('-') && arg.len() > 1 {
            let chars: Vec<char> = arg[1..].chars().collect();
            let mut ci = 0usize;
            let mut consumed_next = false;
            while ci < chars.len() {
                let c = chars[ci];
                match c {
                    'h' => return Ok(ParsedArgs::Help(render_help(command))),
                    'V' if command.is_none() => return Ok(ParsedArgs::Version(version_text())),
                    _ => {}
                }
                let flag = EXTENSION_FLAGS
                    .iter()
                    .chain(GLOBAL_FLAGS.iter())
                    .chain(
                        spec_for(command)
                            .map(|s| s.flags.iter())
                            .unwrap_or([].iter()),
                    )
                    .find(|f| f.matches_short(c))
                    .ok_or_else(|| CliError::UnknownFlag(format!("-{c}")))?;

                match flag.kind {
                    ArgKind::Switch => {
                        apply_switch(flag, &mut verbose, &mut quiet, &mut io);
                        ci += 1; // combined shorts like -vq keep parsing here
                    }
                    kind => {
                        // Value shorts take the rest of this token (-ohello)
                        // or the following one (-o hello).
                        let remainder: String = chars[ci + 1..].iter().collect();
                        let value = if remainder.is_empty() {
                            let next = args
                                .get(i + 1)
                                .ok_or_else(|| CliError::MissingValue(format!("-{c}")))?;
                            if next.starts_with('-')
                                && !matches!(kind, ArgKind::OptionalValue { .. })
                            {
                                return Err(CliError::MissingValue(format!("-{c}")));
                            }
                            consumed_next = true;
                            next.clone()
                        } else {
                            remainder.trim_start_matches('=').to_owned()
                        };
                        let _ = kind;
                        store_value(flag, &value, &mut io, &mut defines)?;
                        break; // whole token consumed
                    }
                }
            }
            i += if consumed_next { 2 } else { 1 };
            continue;
        }

        // Positional: first is the subcommand (or run-target fallback).
        if command.is_none() {
            if arg == "help" {
                // `rubyc help <command>` prints that command's full help.
                let target = args.get(i + 1).and_then(|w| find_command(w));
                return Ok(ParsedArgs::Help(render_help(
                    target.map(|s| command_from_name(s.name)),
                )));
            }
            match find_command(&arg) {
                Some(spec) => {
                    command = Some(command_from_name(spec.name));
                    positionals_left = positionals_for(command.unwrap());
                    i += 1;
                    continue;
                }
                None => {
                    // Bare file path: implicit `run`.
                    command = Some(Command::Run);
                    positionals_left = positionals_for(Command::Run);
                    if positional_is_file(Command::Run) {
                        io.file = Some(arg.clone());
                        positionals_left -= 1;
                    } else {
                        io.file = Some(arg.clone());
                    }
                    i += 1;
                    continue;
                }
            }
        }

        if positionals_left == 0 {
            return Err(CliError::DuplicatePositional(arg));
        }
        let cmd = command.unwrap();
        if positional_is_file(cmd) {
            if io.file.is_some() || io.input.is_some() {
                return Err(CliError::DuplicatePositional(arg));
            }
            io.file = Some(arg);
        } else {
            // Package-name commands: store in io.file as the package name.
            if io.file.is_some() {
                return Err(CliError::DuplicatePositional(arg));
            }
            io.file = Some(arg);
        }
        positionals_left -= 1;
        i += 1;
    }

    let command = command.unwrap_or(Command::CheckRoundtrip);
    Ok(ParsedArgs::Run(Invocation {
        command,
        io,
        verbose,
        quiet,
        error_format,
        no_color,
        lint_overrides,
        defines,
    }))
}

