# Step 25 (optional): North-east monsoon (October–December) extension

**Type**: data + evaluation prompt (mostly configuration).
**Plan items**: C7.
**Depends on**: step 06 (per-(source, year) builders) and step 07 (the
v2 evaluation pipeline).

## Goal

The pitch cites Chennai floods, but those are **north-east monsoon**
(October–December) events, and WEAVR covers only June–September. Both
GraphCast WeatherBench 2 windows span October–December (2017-11-16 →
2019-01-31, and 2019-11-16 → 2021-01-31), HRES and IFS-ENS cover all years,
and IMD observations are year-round.

Build October–December 2018 and 2020 stores and run the tiers. That adds a
real **"season" dimension** to the weights (the PS asks for weights by
season) and makes the south-east India motivation legitimate.

## Prompt

```
Follow Improvements/prompts/README.md's conventions. In WEAVR/:

1. Check coverage: confirm, live, that every source covers 2018-10-01 ..
   2018-12-31 and 2020-10-01 .. 2020-12-31 (with the lagged +/-48 h
   margins). Note that the IMD SEEPS climatology store is JJAS-only, so
   build an OND equivalent with build_seeps_climatology.py's pattern.
   Measure the costs (convention 8).

2. Build OND 2018 and OND 2020 daily stores with step 06's builders
   (a season parameter rather than new scripts).

3. Region scheme: Sreekala & Babu's six zones are SUMMER-monsoon
   homogeneous rainfall zones. Check the paper (research/ has the
   citation) for whether they apply to OND. If not, route via
   AskUserQuestion: reuse the six zones (flagged as a mismatch) / use
   IMD's meteorological subdivisions / a south-peninsula-focused
   grouping.

4. Run the v2 tier scripts with a --season OND option (LOYO 2018 vs
   2020): single-source, independence, tier0, tier1, tier2, scorecard.
   Report the per-season weight differences (JJAS vs OND) per region x
   lead: the "weights by season" result.

5. docs/northeast-monsoon-results.md (plain; including where there are
   too few OND events to fit anything), README, and an "OND" season
   selector in the weight-map and skill-trends views (API + JS; browser-
   verify). If a Chennai / Tamil Nadu event falls within these seasons,
   list it as a replay candidate for step 19's "judge's choice" rather
   than building a new featured replay.
```

## Done when

- October–December 2018/2020 are built and evaluated under LOYO.
- Weights-by-season are reported.
- The dashboard can switch season.
- The south-east India motivation in the pitch is backed by real results
  (or dropped).
