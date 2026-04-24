"""
Detainee Movement Analysis
Deportation Data Project — Stay-Level Detentions Data

Preprocessing:
  Using detention stays data, join country category and category


Movement Analysis:
 Need:
    summary_stats.csv             — for headline numbers
    table1_state_flow.csv         — state × state transfer counts by year
    table2_aor_flow.csv           — AOR × AOR transfer counts by year
    table3_top_facility_pairs.csv — top facility-to-facility routes
    table4_city_flow.csv          — city × city transfer counts by year
    table5_target_states.csv      — transfers INTO target states by year (2023+)
    table6_code12_heatmap_points.csv — ★ NEW Code 1 and Code 2 weighted facility points for ArcGIS heatmap
"""

import os
import pandas as pd


DETENTIONS_FILE  = "/Users/felishawhite/PycharmProjects/immig26/detention-stays-latest 23-25.xlsx"
FACILITIES_FILE  = "/Users/felishawhite/PycharmProjects/immig26/unduplicated_facility_data_immig.xlsx"
CATEGORY_FILE     = "/Users/felishawhite/PycharmProjects/immig26/categ_file.xlsx"
GIS_FACILITY_FILE = "path/to/results/gis/gis_facility_all.csv"  # produced by build_gis_files.py
OUTPUT_DIR        = "/Users/felishawhite/PycharmProjects/immig26/deten-output"
TOP_N            = 30       # number of top facility pairs to return in Table 3




# Detention stays columns
STAY_ID             = "stay_id"
N_STINTS            = "n_stints"
FACILITY_CODE_FIRST = "detention_facility_code_first"
FACILITY_CODE_LAST  = "detention_facility_code_last"
STAY_BOOK_IN        = "stay_book_in_date_time"
CITIZENSHIP_COL     = "citizenship_country"

# Category lookup columns
CAT_COUNTRY  = "country"
CAT_CATEGORY = "Category"
CAT_CODE     = "Category Code"

# Facility lookup columns
FAC_CODE   = "detention_facility_code"
FAC_NAME   = "detention_facility"
FAC_STATE  = "facility_state"
FAC_AOR    = "facility_aor"
FAC_CITY   = "facility_city"
FAC_COUNTY = "facility_county"


# The ten destination states to analyze. Values must match the facility lookup
# exactly — check your facility XLSX for the exact state name format used.

TARGET_STATES = [
    "PENNSYLVANIA",
    "MICHIGAN",
    "WISCONSIN",
    "GEORGIA",
    "ARIZONA",
    "NEVADA",
    "NORTH CAROLINA",
    "TEXAS",
    "FLORIDA",
    "NEW YORK",
    "MINNESOTA",        # added
]

# The five target counties
TARGET_COUNTIES = [
    "MARICOPA COUNTY",
    "GWINNETT COUNTY",
    "COBB COUNTY",
    "OAKLAND COUNTY",
    "BUCKS COUNTY",
]


TARGET_YEAR_START = 2023

# Load main detention stays file and the category lookup file.
# Match each row's citizenship_country to the lookup and inserts two new
# columns — "category" and "category_code" —
# citizenship_country.

def preprocess_detentions(detentions_path, category_path, output_dir):
    print("Step 1: Preprocessing — loading detention stays XLSX (this may take a moment)...")
    df = pd.read_excel(detentions_path, dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    print(f"  Rows loaded: {len(df):,}")

    # Load category lookup and normalize country name for matching
    cat = pd.read_excel(category_path, dtype=str)
    cat.columns = cat.columns.str.strip()
    cat[CAT_COUNTRY]  = cat[CAT_COUNTRY].str.strip().str.upper()
    cat[CAT_CATEGORY] = cat[CAT_CATEGORY].str.strip()
    cat[CAT_CODE]     = cat[CAT_CODE].str.strip()

    # Normalize citizenship_country in the main file to match lookup casing
    df[CITIZENSHIP_COL] = df[CITIZENSHIP_COL].str.strip().str.upper()

    # Merge category columns onto the main dataframe via country name
    df = df.merge(
        cat[[CAT_COUNTRY, CAT_CATEGORY, CAT_CODE]],
        left_on=CITIZENSHIP_COL,
        right_on=CAT_COUNTRY,
        how="left"
    ).drop(columns=[CAT_COUNTRY])

    # Rename merged columns
    df = df.rename(columns={CAT_CATEGORY: "category", CAT_CODE: "category_code"})

    # Reorder: insert "category" and "category_code" immediately after citizenship_country
    cols = list(df.columns)
    cols.remove("category")
    cols.remove("category_code")
    insert_at = cols.index(CITIZENSHIP_COL) + 1
    cols = cols[:insert_at] + ["category", "category_code"] + cols[insert_at:]
    df = df[cols]

    # How many rows mmatch
    matched   = df["category"].notna().sum()
    unmatched = df["category"].isna().sum()
    print(f"  Matched to category:           {matched:,}")
    print(f"  Unmatched (no category found): {unmatched:,}")
    if unmatched > 0:
        print("  Unmatched country values (top 10):")
        print(df[df["category"].isna()][CITIZENSHIP_COL].value_counts().head(10).to_string())

    # Save
    enriched_path = os.path.join(output_dir, "detention_stays_enriched.csv")
    df.to_csv(enriched_path, index=False)
    print(f"  Saved enriched file → {enriched_path}\n")
    return enriched_path



# Include facility_county if that column exists in the file; otherwise skips it.CHATGPT

def load_facilities(path):
    fac = pd.read_excel(path, dtype=str)
    fac.columns = fac.columns.str.strip()
    fac[FAC_CODE]  = fac[FAC_CODE].str.strip().str.upper()
    fac[FAC_AOR]   = fac[FAC_AOR].str.strip().str.title()
    fac[FAC_STATE] = fac[FAC_STATE].str.strip().str.upper()

    # Include county column only if it exists in the file
    base_cols = [FAC_CODE, FAC_NAME, FAC_STATE, FAC_AOR, FAC_CITY]
    if FAC_COUNTY in fac.columns:
        fac[FAC_COUNTY] = fac[FAC_COUNTY].str.strip().str.upper()
        base_cols.append(FAC_COUNTY)

    return fac[base_cols].drop_duplicates(FAC_CODE)


# Chat GPT
# Reads the preprocessed CSV produced in Step 1, normalizes column names,
# and extracts a calendar year from the stay book-in date.

def load_detentions(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df["year"] = pd.to_datetime(df[STAY_BOOK_IN], errors="coerce").dt.year
    df[FACILITY_CODE_FIRST] = df[FACILITY_CODE_FIRST].str.strip().str.upper()
    df[FACILITY_CODE_LAST]  = df[FACILITY_CODE_LAST].str.strip().str.upper()
    df[N_STINTS]            = pd.to_numeric(df[N_STINTS], errors="coerce")
    return df



# A transfer occurred when n_stints > 1, meaning the detainee moved through
# more than one facility during their stay. D

def filter_transfers(df):
    df = df.drop_duplicates(subset=STAY_ID)
    transferred = df[
        (df[N_STINTS] > 1) &
        df[FACILITY_CODE_FIRST].notna() &
        df[FACILITY_CODE_LAST].notna()  &
        (df[FACILITY_CODE_FIRST] != df[FACILITY_CODE_LAST])
    ].copy()
    return transferred


# Joins the facility lookup table twice — once for origin (first code) and
# once for destination (last code). Columns are prefixed "origin_" or "dest_".

def attach_facility_info(df, fac):
    origin = fac.add_prefix("origin_")
    dest   = fac.add_prefix("dest_")
    df = df.merge(origin, left_on=FACILITY_CODE_FIRST,
                  right_on="origin_detention_facility_code", how="left")
    df = df.merge(dest,   left_on=FACILITY_CODE_LAST,
                  right_on="dest_detention_facility_code",   how="left")
    return df


# Groups by the specified columns and returns a total count plus one column
# per calendar year. Sorted descending by total transfers.

def pivot_years(df, group_cols):
    total = df.groupby(group_cols, dropna=False).size().reset_index(name="total_transfers")
    years = (
        df.groupby(group_cols + ["year"], dropna=False)
          .size()
          .unstack("year", fill_value=0)
          .reset_index()
    )
    years.columns = (group_cols +
                     [f"year_{int(y)}" for y in years.columns[len(group_cols):]])
    return total.merge(years, on=group_cols).sort_values("total_transfers", ascending=False)



# Inserts "pct_of_transfers" immediately after "total_transfers".

def add_pct(df):
    total = df["total_transfers"].sum()
    df.insert(df.columns.get_loc("total_transfers") + 1,
              "pct_of_transfers",
              (df["total_transfers"] / total * 100).round(1))
    return df


# Create output folder if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocess main file and add category columns
enriched_csv = preprocess_detentions(DETENTIONS_FILE, CATEGORY_FILE, OUTPUT_DIR)

# Movement analysis
print("Step 2: Movement analysis...")

fac = load_facilities(FACILITIES_FILE)
df  = load_detentions(enriched_csv)
print(f"  Rows loaded:       {len(df):,}")
print(f"  Unique stays:      {df[STAY_ID].nunique():,}")

# Filter to transferred stays (n_stints > 1, deduplicated by stay_ID)
moved        = filter_transfers(df)
total_unique = df.drop_duplicates(subset=STAY_ID)
print(f"  Stays transferred: {len(moved):,}  "
      f"({len(moved) / len(total_unique) * 100:.1f}% of unique stays)")

# Join origin and destination facility metadata
moved = attach_facility_info(moved, fac)

# Summary statistics
cross_state = (moved["origin_facility_state"] != moved["dest_facility_state"]).sum()
cross_aor   = (moved["origin_facility_aor"]   != moved["dest_facility_aor"]).sum()

summary = pd.DataFrame([{
    "total_unique_stays":    len(total_unique),
    "stays_transferred":     len(moved),
    "pct_transferred":       round(len(moved) / len(total_unique) * 100, 1),
    "cross_state_transfers": int(cross_state),
    "pct_cross_state":       round(cross_state / len(moved) * 100, 1),
    "cross_aor_transfers":   int(cross_aor),
    "pct_cross_aor":         round(cross_aor / len(moved) * 100, 1),
    "years_covered":         str(sorted(moved["year"].dropna().unique().astype(int).tolist())),
}]).T.rename(columns={0: "value"})

# Table 1: State to State flow
# How many detainees moved between each pair of states, by year.
t1 = add_pct(pivot_years(moved, ["origin_facility_state", "dest_facility_state"]))

# Table 2: AOR to AOR flow
# Same as Table 1 but at the ICE Area of Responsibility level.
t2 = add_pct(pivot_years(moved, ["origin_facility_aor", "dest_facility_aor"]))

# Table 3: Top facility pairs
# Most common specific origin to destination routes with full location context.
t3 = add_pct(pivot_years(moved, [
    "origin_detention_facility_code", "origin_detention_facility",
    "origin_facility_city",           "origin_facility_state", "origin_facility_aor",
    "dest_detention_facility_code",   "dest_detention_facility",
    "dest_facility_city",             "dest_facility_state",   "dest_facility_aor",
]).head(TOP_N))

# Table 4: City toCity flow
# State is included alongside city to distinguish same-named cities across states.
t4 = add_pct(pivot_years(moved, [
    "origin_facility_city", "origin_facility_state",
    "dest_facility_city",   "dest_facility_state",
]))


# ★  TABLE 5: Transfers INTO target states (2023 onward)  CHATGPT★
#
# Filter where the DESTINATION facility is in one of the ten
# target states and the stay began in TARGET_YEAR_START or later.
# Groups by destination state + year so you can compare volume across
# states and track change over time in a single combined table.
#

# Filter to target states and year range
target = moved[
    (moved["dest_facility_state"].isin(TARGET_STATES)) &
    (moved["year"] >= TARGET_YEAR_START)
].copy()

print(f"\n  Target state transfers (2023+): {len(target):,}")

# State-level summary: transfers into each target state by year
t5_state = add_pct(pivot_years(target, ["dest_facility_state"]))

# City-level breakdown within target states
t5_city = add_pct(pivot_years(target, [
    "dest_facility_state", "dest_facility_city"
]))

# Origin breakdown: where are detainees coming FROM into each target state?
t5_origin = add_pct(pivot_years(target, [
    "origin_facility_state", "dest_facility_state"
]))



# ════════════════════════════════════════════════════════════════════════════
# ★  TABLE 6: Code 1 and Code 2 weighted facility points for ArcGIS heatmap  ★
#
# Filters transferred stays to Code 1 (African Countries) and Code 2
# (African Diaspora) detainees only, aggregates by destination facility,
# then joins geocoded lat/lon coordinates from gis_facility_all.csv.
# The output is a point file weighted by transfer volume — ready to load
# into ArcGIS Online as a heatmap layer or proportional symbol layer.
#
# REQUIREMENT: Run build_gis_files.py first to produce gis_facility_all.csv.
#   If that file doesn't exist yet this table will be skipped automatically.
# ════════════════════════════════════════════════════════════════════════════

if os.path.exists(GIS_FACILITY_FILE):
    print("\n  Building Table 6: Code 1 and Code 2 heatmap points...")

    # Reload enriched file with category_code included
    enriched = pd.read_csv(
        os.path.join(OUTPUT_DIR, "detention_stays_enriched.csv"),
        dtype=str, low_memory=False
    )
    enriched.columns = enriched.columns.str.strip().str.lower()
    enriched["category_code"] = enriched["category_code"].str.strip()
    enriched["n_stints"]      = pd.to_numeric(enriched["n_stints"], errors="coerce")
    enriched[FACILITY_CODE_LAST] = enriched[FACILITY_CODE_LAST].str.strip().str.upper()

    # Filter to Code 1 and Code 2 transferred stays in target states only
    code12 = enriched[
        enriched["category_code"].isin(["1", "2"]) &
        (enriched["n_stints"] > 1) &
        enriched[FACILITY_CODE_LAST].notna()
    ].drop_duplicates(subset=STAY_ID).copy()

    print(f"    Code 1 and Code 2 transferred stays: {len(code12):,}")

    # Aggregate transfer counts by destination facility code and category
    t6_agg = (
        code12.groupby([FACILITY_CODE_LAST, "category_code"], dropna=False)
              .size()
              .reset_index(name="transfer_count")
    )

    # Add category label for ArcGIS legend
    t6_agg["category_label"] = t6_agg["category_code"].map({
        "1": "African Country",
        "2": "African Diaspora Country",
    })

    # Load geocoded facility file and join coordinates
    gis_fac = pd.read_csv(GIS_FACILITY_FILE, dtype=str, low_memory=False)
    gis_fac.columns = gis_fac.columns.str.strip().str.lower()
    gis_fac["detention_facility_code"] = gis_fac["detention_facility_code"].str.strip().str.upper()

    t6 = t6_agg.merge(
        gis_fac[[
            "detention_facility_code", "detention_facility",
            "facility_city", "facility_state", "facility_aor",
            "latitude", "longitude", "match_status"
        ]],
        left_on=FACILITY_CODE_LAST,
        right_on="detention_facility_code",
        how="left"
    ).drop(columns=["detention_facility_code"])

    # Convert transfer count to numeric for ArcGIS weighting
    t6["transfer_count"] = pd.to_numeric(t6["transfer_count"], errors="coerce")

    # Flag records missing coordinates — these need manual geocoding in ArcGIS
    missing_coords = t6["latitude"].isna().sum()
    print(f"    Facilities with coordinates:         {t6['latitude'].notna().sum():,}")
    print(f"    Facilities missing coordinates:      {missing_coords:,}  (need ArcGIS manual geocoding)")

    t6.to_csv(os.path.join(OUTPUT_DIR, "table6_code12_heatmap_points.csv"), index=False)
    print(f"    Saved → table6_code12_heatmap_points.csv  ★ NEW")
else:
    print("\n  Skipping Table 6 — gis_facility_all.csv not found.")
    print("  Run build_gis_files.py first, then re-run this script.")


# ════════════════════════════════════════════════════════════════════════════
# ★  TABLE 7: Code 1 and Code 2 state-to-state flow  CHAT GPT★
#
# Filters transferred stays to Code 1 (African Countries) and Code 2
# (African Diaspora) detainees only, then produces a state-to-state flow
# table in the same structure as table5c_target_origin_flow.csv.
# This allows geographic pipeline findings to speak specifically to
# Black immigrant movement rather than total detainee movement.
#
# Output columns: origin_facility_state, dest_facility_state,
#   category_code, category_label, total_transfers, pct_of_transfers,
#   and one column per calendar year.
# ════════════════════════════════════════════════════════════════════════════

print("\n  Building Table 7: Code 1 and Code 2 state-to-state flow...")

# category_code is already present in "moved" because it was loaded from
# detention_stays_enriched.csv in Step 1 — no reload or merge needed.
# Normalize the column and filter directly.
moved["category_code"] = moved["category_code"].astype(str).str.strip()

# Filter to Code 1 and Code 2 only
code12_flow = moved[
    moved["category_code"].isin(["1", "2"])
].copy()

print(f"    Code 1 and Code 2 transferred stays with state data: {len(code12_flow):,}")

# Add human-readable category label
code12_flow["category_label"] = code12_flow["category_code"].map({
    "1": "African Country",
    "2": "African Diaspora Country",
})

# Aggregate: origin state → destination state, by category and year
# Produces the same pivot_years structure as all other flow tables
t7 = add_pct(pivot_years(code12_flow, [
    "origin_facility_state",
    "dest_facility_state",
    "category_code",
    "category_label",
]))

# Also produce a combined (Code 1 + Code 2 together) state flow
# for simpler mapping when category breakdown is not needed
t7_combined = add_pct(pivot_years(code12_flow, [
    "origin_facility_state",
    "dest_facility_state",
]))

print(f"    Unique origin → destination state pairs: {len(t7):,}")
print(f"    Combined (Code 1+2) state pairs:         {len(t7_combined):,}")

t7.to_csv(os.path.join(OUTPUT_DIR, "table7a_code12_state_flow_by_category.csv"), index=False)  # ★ NEW
t7_combined.to_csv(os.path.join(OUTPUT_DIR, "table7b_code12_state_flow_combined.csv"),  index=False)  # ★ NEW

# Save all outputs
summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"))
t1.to_csv(os.path.join(OUTPUT_DIR, "table1_state_flow.csv"),              index=False)
t2.to_csv(os.path.join(OUTPUT_DIR, "table2_aor_flow.csv"),                index=False)
t3.to_csv(os.path.join(OUTPUT_DIR, "table3_top_facility_pairs.csv"),      index=False)
t4.to_csv(os.path.join(OUTPUT_DIR, "table4_city_flow.csv"),               index=False)
t5_state.to_csv(os.path.join(OUTPUT_DIR, "table5a_target_state_flow.csv"),  index=False)   # ★ NEW
t5_city.to_csv(os.path.join(OUTPUT_DIR,  "table5b_target_city_flow.csv"),   index=False)   # ★ NEW
t5_origin.to_csv(os.path.join(OUTPUT_DIR,"table5c_target_origin_flow.csv"), index=False)
t6.to_csv(os.path.join(OUTPUT_DIR, "table6_code12_heatmap_points.csv"),       index=False)  # ★ NEW — only saved if Table 6 ran successfully

print(f"\nAll outputs written to: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
