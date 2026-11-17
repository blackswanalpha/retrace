//! Retrace: cost-aware n-gram drafter for speculative decoding.
//!
//! Proposes draft tokens by finding the longest recent n-gram in the token
//! history that matches the current suffix and copying what followed it.
//! The index is updated incrementally, so each call costs O(new tokens).
//!
//! Draft length is chosen per call to maximize expected tokens per unit of
//! compute: it combines a learned per-token acceptance rate for each match
//! length with a measured cost table for verifying k tokens in one pass.
//! When no length beats plain decoding, it drafts nothing.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Prior acceptance: 1 success in 2 trials.
const PRIOR_SUCC: f64 = 1.0;
const PRIOR_TRIALS: f64 = 2.0;
/// After this many skipped matches at one length, probe with a 1-token draft.
const PROBE_EVERY: u32 = 16;

fn hash_ngram(tokens: &[i32]) -> u64 {
    let mut h = FNV_OFFSET;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Expected accepted tokens for a draft of length d with per-token acceptance p.
fn expected_accepted(p: f64, d: usize) -> f64 {
    let mut e = 0.0;
    let mut pk = 1.0;
    for _ in 0..d {
        pk *= p;
        e += pk;
    }
    e
}

#[pyclass]
pub struct Retrace {
    min_n: usize,
    max_n: usize,
    k_max: usize,
    /// costs[j] = time to verify j + 1 tokens in one pass, relative to 1 token.
    costs: Vec<f64>,
    decay: f64,
    history: Vec<i32>,
    /// index[n - min_n] maps an n-gram hash to the position right after its latest occurrence.
    index: Vec<HashMap<u64, usize>>,
    /// Number of history positions whose ending n-grams are indexed.
    indexed: usize,
    /// Decayed acceptance counts per match length.
    succ: Vec<f64>,
    trials: Vec<f64>,
    skips: Vec<u32>,
    last_draft: Vec<i32>,
    last_len: usize,
    last_n: usize,
    calls: u64,
    drafts: u64,
    drafted: u64,
    accepted: u64,
    skipped: u64,
}

impl Retrace {
    fn reset_history(&mut self) {
        self.history.clear();
        for m in &mut self.index {
            m.clear();
        }
        self.indexed = 0;
        self.last_draft.clear();
        self.last_len = 0;
    }

    fn observe(&mut self, n: usize, accepted: usize, drafted: usize) {
        let i = n - self.min_n;
        let tested = if accepted < drafted { accepted + 1 } else { accepted };
        self.succ[i] = self.decay * self.succ[i] + accepted as f64;
        self.trials[i] = self.decay * self.trials[i] + tested as f64;
    }

    /// Bring history in line with `tokens`, learning from the last draft's outcome.
    fn sync(&mut self, tokens: &[i32]) {
        let n_old = self.history.len();
        let continues = tokens.len() >= n_old && tokens[..n_old] == self.history[..];
        if !continues {
            self.reset_history();
            self.history.extend_from_slice(tokens);
            return;
        }
        if !self.last_draft.is_empty() && self.last_len == n_old {
            let acc = tokens[n_old..]
                .iter()
                .zip(&self.last_draft)
                .take_while(|(a, b)| a == b)
                .count();
            self.accepted += acc as u64;
            let (n, d) = (self.last_n, self.last_draft.len());
            self.observe(n, acc, d);
        }
        self.last_draft.clear();
        self.history.extend_from_slice(&tokens[n_old..]);
    }

    /// Index n-grams ending before position `upto` (exclusive of the current suffix).
    fn index_until(&mut self, upto: usize) {
        while self.indexed < upto {
            let end = self.indexed + 1;
            for n in self.min_n..=self.max_n {
                if end < n {
                    break;
                }
                let h = hash_ngram(&self.history[end - n..end]);
                self.index[n - self.min_n].insert(h, end);
            }
            self.indexed += 1;
        }
    }

    /// Longest matching suffix: returns (match length, continuation start).
    fn find_match(&self) -> Option<(usize, usize)> {
        let len = self.history.len();
        for n in (self.min_n..=self.max_n).rev() {
            if len < n + 1 {
                continue;
            }
            let suffix = &self.history[len - n..];
            if let Some(&start) = self.index[n - self.min_n].get(&hash_ngram(suffix)) {
                // Guard against hash collisions.
                if start >= n && start < len && self.history[start - n..start] == *suffix {
                    return Some((n, start));
                }
            }
        }
        None
    }

    /// Draft length maximizing (1 + expected accepted) / cost, or 0 if none beats plain decoding.
    fn choose_len(&self, n: usize, available: usize) -> usize {
        let i = n - self.min_n;
        let p = self.succ[i] / self.trials[i];
        let d_max = available.min(self.k_max).min(self.costs.len() - 1);
        let (mut best_d, mut best) = (0, 1.0);
        for d in 1..=d_max {
            let gain = (1.0 + expected_accepted(p, d)) / self.costs[d];
            if gain > best {
                best = gain;
                best_d = d;
            }
        }
        best_d
    }
}

#[pymethods]
impl Retrace {
    #[new]
    #[pyo3(signature = (min_n=2, max_n=4, k_max=8, costs=None, decay=0.95))]
    fn new(min_n: usize, max_n: usize, k_max: usize, costs: Option<Vec<f64>>, decay: f64) -> PyResult<Self> {
        if min_n == 0 || max_n < min_n {
            return Err(PyValueError::new_err("need 1 <= min_n <= max_n"));
        }
        if k_max == 0 {
            return Err(PyValueError::new_err("need k_max >= 1"));
        }
        if !(0.0 < decay && decay <= 1.0) {
            return Err(PyValueError::new_err("need 0 < decay <= 1"));
        }
        // Default: each extra verified token costs 40% of a single-token pass.
        let costs = costs.unwrap_or_else(|| (0..=k_max).map(|j| 1.0 + 0.4 * j as f64).collect());
        if costs.len() < 2 || costs.iter().any(|&c| c <= 0.0) {
            return Err(PyValueError::new_err("costs needs >= 2 positive entries, costs[0] for 1 token"));
        }
        let levels = max_n - min_n + 1;
        Ok(Self {
            min_n,
            max_n,
            k_max,
            costs,
            decay,
            history: Vec::new(),
            index: (0..levels).map(|_| HashMap::new()).collect(),
            indexed: 0,
            succ: vec![PRIOR_SUCC; levels],
            trials: vec![PRIOR_TRIALS; levels],
            skips: vec![0; levels],
            last_draft: Vec::new(),
            last_len: 0,
            last_n: min_n,
            calls: 0,
            drafts: 0,
            drafted: 0,
            accepted: 0,
            skipped: 0,
        })
    }

    /// `data` is the full token history as little-endian int32 bytes.
    /// Returns draft tokens, or an empty list when drafting is not worth its cost.
    fn draft(&mut self, data: &[u8]) -> PyResult<Vec<i32>> {
        if data.len() % 4 != 0 {
            return Err(PyValueError::new_err("data length must be a multiple of 4"));
        }
        let tokens: Vec<i32> = data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.calls += 1;
        self.sync(&tokens);
        let len = self.history.len();
        if len < 2 {
            return Ok(Vec::new());
        }
        self.index_until(len - 1);
        let Some((n, start)) = self.find_match() else {
            return Ok(Vec::new());
        };
        let i = n - self.min_n;
        let mut d = self.choose_len(n, len - start);
        if d == 0 {
            self.skips[i] += 1;
            if self.skips[i] < PROBE_EVERY {
                self.skipped += 1;
                return Ok(Vec::new());
            }
            d = 1;
        }
        self.skips[i] = 0;
        let draft = self.history[start..start + d].to_vec();
        self.drafts += 1;
        self.drafted += d as u64;
        self.last_draft = draft.clone();
        self.last_len = len;
        self.last_n = n;
        Ok(draft)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("calls", self.calls)?;
        d.set_item("drafts", self.drafts)?;
        d.set_item("drafted", self.drafted)?;
        d.set_item("accepted", self.accepted)?;
        d.set_item("skipped", self.skipped)?;
        let p: Vec<f64> = self.succ.iter().zip(&self.trials).map(|(s, t)| s / t).collect();
        d.set_item("accept_p_by_n", p)?;
        Ok(d)
    }
}

#[pymodule]
fn retrace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Retrace>()?;
    Ok(())
}
