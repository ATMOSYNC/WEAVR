# WEAVR execution plan — one agent, one sequence

Companion to [`WEAVR-SIH-improvement-plan.md`](WEAVR-SIH-improvement-plan.md)
(what to build and why) and [`prompts/README.md`](prompts/README.md) (the
remaining prompts and their dependencies).

Steps run one at a time, in the order below.

**Done:** 01–05 — see [`prompts/completed/DONE.md`](prompts/completed/DONE.md).
**Remaining:** 21 prompts (06–26): 11 in the sequence below, 6 P1 add-ons, 4 optional.

---

## 1. The sequence

Each row starts when the one above it is merged into `main`. Doc-only PRs
run no CI; everything else needs green lint/test before merge.

| Order | Step | What | Class |
|---|---|---|---|
| 1 | **06** | 2018 season stores (make leave-one-year-out possible) | must |
| 2 | **07** | Regenerate all results under LOYO, with CIs — **the gate** | must |
| 3 | **11** | Tail repair: quantile mapping | must |
| 4 | **12** | Tail repair: extreme-value tail | must |
| 5 | **08** | Integrate NEPS-G (measure its error correlation first) | must* |
| 6 | **09** | Independence weighting and per-model contributions | must |
| 7 | **14** | District warnings science | must |
| 8 | **17** | Dashboard: district warnings and explainer | must |
| 9 | **19** | Kerala 2018 event replay | should |
| 10 | **20** | Static deploy, reproducibility, model card | should |
| 11 | **26** | Pitch rebuild and finale prep | must |
| — | 10, 13, 15, 16, 18, 21 | P1 refinements — slot in where time allows, see §2 | optional-ish |
| — | 22, 23, 24, 25 | Optional models / extensions — see §3 | optional |

\* 08 is in the plan's P0 set but carries a go/no-go: see §3.

Why this order:

- **06 → 07 first.** Eleven steps sit behind 07. It rewrites every file in
  `results/`, so nothing else should be in flight while it is open.
- **11 → 12 before 09.** Step 01 showed the tail (115.6 mm and above) is the
  real gap, and 14 needs the repaired probabilities. 09 blocks nothing on the
  critical path.
- **14 → 17 → 19 → 20** is the product spine; 15, 16 and 18 hang off it.
- **26 last**, and re-run its numbers refresh whenever a late result lands.

## 2. Optional-ish P1 steps: where they fit

| Step | Fits after | Skip if |
|---|---|---|
| 10 live daily pipeline | 07 | you won't demo anything live |
| 13 combine the combiners | 12 | short of time (a refinement of a result you have) |
| 15 trust scale / economic value | 12 | short of time (weakest-sourced claim) |
| 16 dashboard evidence views | 09 and 15 | 17 is done and time is short |
| 18 CAP export | 17 | 17 already shows the warnings |
| 21 self-healing weights | 10 | 10 alone demonstrates a live pipeline |

Careful with 16: it depends on 15 and 09, so skipping 15 also removes its
value API from 16.

## 3. Decision points

Three moments where you should consciously choose, rather than drift:

**After 07 lands — do the extra season and daily cadence actually help?**
07 reports v2 results under LOYO with confidence intervals. If the blends
still lose to raw GraphCast with more data and proper intervals, that is a
real result and the pitch's framing changes (from "our blend wins" to "we
measured exactly when blending helps, and it's at the tail"). Decide the
narrative here, not in week 4.

**After 08's correlation measurement — is NEPS-G worth the full build?**
One number decides it: NEPS-G's error correlation against the existing
three. If it is as collinear as HRES and IFS-ENS are with each other
(0.77–0.93), the effective-models count barely moves and the remaining
integration effort is better spent elsewhere. Measure before committing.

**Before step 14 — which optionals survive?** 22, 23, 24 and 25 cost
16–21 pd between them and none is on the critical path. Pick by what the
team is short of:
- **23 (Tier 4 DRN)** — the answer to "where is your AI?", if a judge asks.
- **24 (temperature)** — covers more of the problem statement.
- **22 (GenCast/FuXi)** — the most direct attack on the N_eff ≈ 1.1 finding.
- **25 (OND)** — only if the Chennai/south-east framing is being reinstated.

---

## 4. If you run short: the descope ladder

Cut in this order. Each line is the cheapest remaining thing whose loss does
the least damage:

1. **25** (OND extension, P2) — the pitch no longer depends on it; step 01
   already removed the Chennai claim.
2. **22, 23, 24** (optional models) — keep at most one, chosen above.
3. **21** (self-healing weights) — 10 alone still demonstrates a live
   pipeline.
4. **18** (CAP export) — nice, but 17 already shows the warnings.
5. **16** (evidence views) — the numbers still exist in the docs; only the
   dashboard presentation is lost.
6. **13** (combine the combiners) — a P1 refinement of a result you already
   have.
7. **15** (trust scale / economic value) — the weakest-sourced claim in the
   set anyway; illustrative bands are easy for a judge to puncture.
8. **20** (static deploy) — only if you're certain the live demo will work.
   Cutting this removes your offline fallback, so cut it last.

**Never cut:** 02–07 (the evidence base), 11, 12 (the tail
repair that step 01 showed is the actual gap), 14, 17 (the product a judge
sees), 26 (the pitch). That set is the submission.

---

## 5. What can go wrong

| Risk | Effect | Response |
|---|---|---|
| **06's fetch is slow or fails** | 07 slips, LOYO impossible. | Measure one chunk first, run resumable in the background, sequential builds only. If it exceeds a day, run 07 on 2020-daily alone, keep `seasonal_block_split`, and say so plainly. The known trap: 2018 GraphCast uses `lat`/`lon`. |
| **07 contradicts a pre-registered claim** | A headline claim fails. | This is the system working. Report it as Tier 3's no-go was reported. |
| **Mixed cadence misleads** | Tier 2 numbers rest on 18 inits, others on 122. | State it beside every Tier 2 number (see `completed/DONE.md`). Pulling the daily IFS-ENS store (126 GB) would remove it; it was skipped. |
| **Degenerate bootstrap** | Zero-width CIs read "significant". | Already handled: `bootstrap_is_degenerate` returns NaN bounds. Don't bypass it. |

---

## 6. One-page summary

`06 → 07 → 11 → 12 → 08 (go/no-go) → 09 → 14 → 17 → 19 → 20 → 26`, with
10, 13, 15, 16, 18, 21 added as time allows and 22–25 only by choice.
Start with **06**.
