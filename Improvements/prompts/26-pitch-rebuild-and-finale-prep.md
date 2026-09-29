# Step 26: Rebuild the pitch from verified numbers, and prepare the finale

**Type**: pitch + rehearsal prompt. Mostly outside the repo (`ppt/`,
`Improvements/`).
**Plan items**: plan §3 (narrative), §8 (demo storyline), §9 (Q&A), §10
(risks), Wave 3.
**Depends on**: every step you chose to complete. Run it last, and re-run
its numbers-refresh part whenever a later result lands.

## Goal

Make the pitch a faithful summary of what the repo proves, then make the
demo and Q&A impossible to derail:

1. Rewrite `ppt/content/` from `results/preregistration_verdicts.csv`
   and the results docs. **Only PASS claims appear as findings.** Failed
   and exploratory results appear as honest limits.
2. Add whatever the official SIH template requires (e.g. a Technical
   Approach slide with an architecture diagram and the tier ladder).
3. Produce a timed demo script, a Q&A drill sheet with real numbers, an
   offline demo bundle, and a finale-day checklist.

## Prompt

```
Work outside the WEAVR repo unless a step says otherwise. Follow
Improvements/prompts/README.md's conventions (measure before claiming;
never overclaim).

1. Fetch the official SIH 2026 PPT template and rubric. Ask the user for
   the file or link if you can't find the official source; don't guess its
   structure. List the required slides and any word/slide limits.

2. Build a claims ledger: Improvements/pitch/claims-ledger.md. One row per
   claim the slides will make, with columns: slide, exact wording, source
   file and row (results CSV / doc + commit hash), CI, and the
   pre-registration id (or "exploratory"). Every number on every slide
   must appear here. Include the one-line findings from steps 01
   (F2/F3), 07, 08, 09, 12, 14, 19, and 13/21/22/23/24 if they were done.

3. Rewrite ppt/content/slide*.md to match the template, using only ledger
   rows:
   - Positioning: the plan's section 3 line, with measured N_eff / NEPS-G
     numbers substituted (or that part removed if the measurement didn't
     support it).
   - A Technical Approach slide: data sources (with years), the tier
     ladder (T0 -> T1 -> T2 / T2b -> tail repair -> T4 if done), the
     independence analysis, and decision products (district warnings, CAP,
     value). Include a simple architecture diagram description, or an SVG
     in ppt/.
   - Proof: the scorecard summary with CIs, the Kerala replay outcome
     (whatever it was), and what failed (Tier 3; anything else).
   - Impact: economic value for real user types (illustrative bands
     labelled as such); district warnings; CAP; adoption path (the
     reproducible build, model card, daily pipeline).
   - Keep the rewrite measured. No "provably", no unverified
     "SACHET-compatible", no NEPS-G claims beyond what step 08 showed.

4. Demo script: Improvements/pitch/demo-script.md, following the plan's
   section 8 storyline, with exact clicks, the view per beat, timings
   adding up to the allotted demo time, and a fallback for each beat
   (static site / screenshots) if something fails live.

5. Q&A drill: Improvements/pitch/qa-drill.md. Take the plan's section 9
   table, replace each prepared answer with the real, measured answer
   and its source row, and add any new hard questions the actual results
   raise (e.g. "why did X fail?"). Mark the 5 most likely questions.

6. Offline demo bundle checklist: Improvements/pitch/finale-checklist.md.
   - the static site build (step 20) on the laptop AND a USB drive
   - local copies of the data stores the live replay needs
   - a cached "today" snapshot (step 10)
   - screenshots of every view
   - the Q&A sheet, printed
   - chargers and adapters
   - a no-internet rehearsal: run the full demo with Wi-Fi off, and record
     what broke
   - a "built at the venue" log template (to separate pre-built from
     venue work, if asked)
   - a shortlist of small, visible features to build live with mentors
     (e.g. a bilingual template bulletin from district warnings,
     extra judge's-choice presets, CAP polish)

7. Rehearse: time two full dry runs of the demo script against the running
   static site. Note every stumble in the checklist, and fix or cut it.

8. Inside the repo (only if needed): fix any README / doc sentence the
   ledger found out of sync with the results. One small PR.
```

## Done when

- Every slide claim traces to a ledger row.
- The deck matches the official template.
- The demo script is timed and rehearsed twice, offline included.
- The Q&A drill has real, sourced answers.
- The finale checklist is complete.
