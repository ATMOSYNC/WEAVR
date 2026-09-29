# Step 19: Event replay — Kerala, August 2018 (plus "judge's choice")

**Type**: evaluation + frontend prompt. This is the pitch centrepiece.
**Plan items**: D1, and pre-registered H9.
**Depends on**: step 07 (LOYO models), step 08 (NEPS-G), step 14
(district warnings) and step 17 (district view components).

## Goal

A replay view showing what WEAVR would have said **before a real disaster**,
using a model trained **without that season**:

- **Primary event:** the Kerala floods of August 2018. That month is
  covered by GraphCast (2018 window), HRES, IFS-ENS, NEPS-G (day 1) and
  IMD. **Models are trained on 2020 only**, so the replay is genuinely
  out-of-sample.
- **Secondary event:** the user's choice from those the data supports
  (e.g. Mumbai, early July 2019: HRES, IFS-ENS and NEPS-G, no GraphCast;
  or a 2020 event, using the 2018-trained fold).
- **"Judge's choice":** any JJAS day in 2018 / 2019 / 2020 can be replayed
  live, which shows the featured event wasn't cherry-picked.

For each featured event, report the **first lead time at which WEAVR would
have issued Orange and Red** over the affected districts, next to each raw
model's deterministic colour and raw IFS-ENS probabilities.

## Integrity rules

- **Event dates and affected districts come from the data, fixed before
  any forecast is looked at.** Identify the peak days and districts from
  IMD observations (e.g. district exceedance fractions from step 14 over
  Kerala in August 2018). Commit that definition as its own commit, before
  running the replay. That is H9's pre-registration requirement.
- Don't tune anything on the event. The replay uses exactly the
  already-fitted LOYO models and the step-14 colour rule.
- Show the result whatever it is. If WEAVR's warning comes late or never,
  the view says so, and the doc explains why (e.g. via the tail analysis
  from step 12). An honest miss is still a strong slide; a tuned hit is a
  liability.
- Label everything "hindcast replay", never "WEAVR predicted".

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Route the secondary-event choice via AskUserQuestion. List 2-4
   candidates the data actually supports (check source availability per
   date; don't recall it from memory), each with its sources and the
   training fold it would use.

2. Event definition, committed before any replay run:
   - For each featured event, add scripts/define_replay_events.py writing
     docs/replay-events.csv: event id, IMD days, affected districts
     (a documented rule, e.g. district exceedance fraction >= 25% at
     >= 115.6 mm on a peak day, using step 14's functions on IMD obs
     only), and the training fold to use.
   - Commit this alone first.

3. Add scripts/export_event_replay.py:
   - For each featured event, and each lead 120 -> 24 h verifying on each
     event day, write:
     - WEAVR district probabilities and colours (LOYO model; step-14 rule)
     - raw deterministic colours for each available source
     - raw IFS-ENS member-counting probabilities
     - NEPS-G (24 h, where available)
     - IMD observed district categories
   - Output dashboard/data/replay_<event_id>.json.
   - Judge's choice: a compact district-level JSON for EVERY JJAS day of
     2018/2019/2020 (WEAVR colour + observed category per district per
     lead). Measure the file size first; if it exceeds ~20 MB, split it
     per season and lazy-load.
   - Summarise into results/replay_summary.csv: first lead with Orange and
     Red per method per event, hits/misses over the affected districts.
   - Mark HEPPI-derived fields per step 08's licence decision (omit them
     from anything the static site will publish if that was the
     decision).

4. API: /api/replay/events, /api/replay?event_id=&lead= and
   /api/replay-day?date=&lead= (422 on bad params; tested).

5. Frontend: dashboard-web/js/charts/eventReplay.js (module-prefixed
   globals), reusing districtWarnings.js's projection/path code (factor it
   into a shared, prefixed helper if needed; don't duplicate it). Layout:
   - event picker, plus a date picker for judge's choice
   - a lead slider stepping from day 5 to day 1
   - three synchronised maps: WEAVR | a raw-model selector | IMD observed
   - a summary strip: "first Orange at day X, first Red at day Y" per
     method
   - caption: hindcast replay, training fold, sources available, and the
     honest outcome from docs/event-replay-results.md
   Browser-verify both events at every lead, and 3 random judge's-choice
   days across all three seasons. Check that the maps match the API JSON
   and that there are no console errors; check mobile width.

6. docs/event-replay-results.md: the event definitions (with the commit
   hash proving they came first), the per-lead tables and the H9 report,
   in plain language. README paragraph. One PR, with the pre-committed
   event-definition commit visible in its history.
```

## Done when

- The Kerala 2018 and secondary replays exist, built from pre-committed
  event definitions and out-of-sample models.
- "Judge's choice" works for any JJAS day of 2018–2020.
- H9 is reported whatever the outcome.
- The view is verified in a real browser.
