//! "Did you mean …?" suggestion engine.
//!
//! Levenshtein distance ≤ 2 (with case-insensitive and common-prefix
//! bonuses) picks the closest candidate from a provided set; the result is
//! rendered as the diagnostic's `help:` line.

/// Classic DP edit distance (insert/delete/substitute), early-capped.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Pick the closest candidate within threshold `max`, preferring:
/// 1. exact match (returns it directly),
/// 2. case-insensitive equality,
/// 3. smallest edit distance (ties → shorter candidate → lexicographic).
pub fn suggest<'a>(name: &str, candidates: &[&'a str], max: usize) -> Option<&'a str> {
    if candidates.is_empty() {
        return None;
    }
    if let Some(exact) = candidates.iter().find(|c| **c == name) {
        return Some(exact);
    }
    let lower = name.to_ascii_lowercase();
    if let Some(ci) = candidates.iter().find(|c| c.to_ascii_lowercase() == lower) {
        return Some(ci);
    }
    let mut best: Option<(usize, &'a str)> = None;
    for cand in candidates {
        let d = edit_distance(name, cand);
        if d == 0 || d > max {
            continue;
        }
        // Prefer smaller distance; on ties prefer candidate that starts
        // with the typed prefix, then the shorter name.
        let better = match best {
            None => true,
            Some((bd, bc)) => {
                d < bd
                    || (d == bd
                        && ((cand.starts_with(name) && !bc.starts_with(name))
                            || cand.len() < bc.len()))
            }
        };
        if better {
            best = Some((d, cand));
        }
    }
    best.map(|(_, c)| c)
}

/// Rendered `did you mean 'x'?` help text, or None when nothing is close.
pub fn did_you_mean(name: &str, candidates: &[&str]) -> Option<String> {
    suggest(name, candidates, 2).map(|c| format!("did you mean '{c}'?"))
}
