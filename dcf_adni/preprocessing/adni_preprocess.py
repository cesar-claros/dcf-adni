"""
ADNI Data Preprocessing Module.

Provides the ADNIPreprocess class that loads raw ADNI clinical and MRI data,
encodes multi-hot variables, filters cognitively-normal (CN) subjects into
transition vs. no-transition cohorts, performs propensity-matched pairing,
applies variance-based feature selection, and exports the final datasets.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

log = logging.getLogger(__name__)


class ADNIPreprocess:
    """End-to-end preprocessing pipeline for ADNI data.

    Parameters
    ----------
    data_path : str
        Path to the main ADNI subjects CSV.
    mri_path : str
        Path to the FreeSurfer MRI CSV (UCSFFSX7).
    output_dir : str
        Directory where output CSVs will be saved.
    variance_threshold : float
        Minimum variance required to keep a feature.
    """

    # Multi-hot variables to one-hot encode
    MULTIHOT_VARS = [
        "DXMOTHET", "KEYMED", "PTNLANG", "MHNUM",
        "PTMARRY", "PTHOME", "PTNOTRT", "PTPLANG",
    ]

    # Columns that need numeric coercion
    NUMERIC_COERCE_COLS = [
        "RCT20", "RCT1408", "RCT19", "HMT40", "BAT126", "RCT392", "RCT14",
    ]

    # Categorical / metadata columns excluded from variance filtering
    CAT_VARS = [
        "subject_id", "visit", "group", "transition", "GENOTYPE",
        "research_group", "subject_date", "KEYMED", "PTNLANG", "DXMOTHET",
    ]

    # Variables used for cohort matching
    NUM_MATCH_VARS = ["subject_age"]
    CAT_MATCH_VARS = ["PTGENDER", "GENOTYPE"]

    # Timepoint labels
    TIMEPOINTS_BASELINE = ["sc", "bl"]
    TIMEPOINTS_ALL = ["sc", "bl", "4_sc", "4_bl"]

    def __init__(
        self,
        data_path: str = "data/All_Subjects_My_Table_03Jul2025.csv",
        mri_path: str = "data/UCSFFSX7_20Jun2025.csv",
        output_dir: str = "data/",
        variance_threshold: float = 0.05,
    ):
        self.data_path = data_path
        self.mri_path = mri_path
        self.output_dir = output_dir
        self.variance_threshold = variance_threshold

        # Intermediate dataframes populated by pipeline steps
        self.data_df: pd.DataFrame | None = None
        self.mri_df: pd.DataFrame | None = None
        self.subjects_transition_df: pd.DataFrame | None = None
        self.subjects_no_transition_df: pd.DataFrame | None = None
        self.subjects_pairs_df: pd.DataFrame | None = None
        self.joint_dataset_df: pd.DataFrame | None = None
        self.remaining_test_df: pd.DataFrame | None = None
        self.keep_features: list[str] | None = None

        # Data-type mapping for ADNI columns
        self.datatypes_dict = {
            "ABETA42": float, "BAT126": float, "CV": float,
            "DILUTION_CORRECTED_CONC": float, "PTAU": float,
            "RCT14": float, "RCT392": float, "TAU": float,
            "BCPREDX": "Int64", "CDCARE": "Int64", "CDHOME": float,
            "CDRSB": float, "DXDSEV": "Int64", "DXMDUE": "Int64",
            "DXMOTHET": str,
            "DXMOTHET_1": "Int64", "DXMOTHET_2": "Int64",
            "DXMOTHET_4": "Int64", "DXMOTHET_5": "Int64",
            "DXMOTHET_6": "Int64", "DXMOTHET_7": "Int64",
            "DXMOTHET_9": "Int64", "DXMOTHET_11": "Int64",
            "DXMOTHET_12": "Int64", "DXMOTHET_14": "Int64",
            "DXMPTR1": "Int64", "DXMPTR2": "Int64",
            "DXMPTR3": "Int64", "DXMPTR4": "Int64", "DXMPTR5": "Int64",
            "FAQFINAN": "Int64", "FAQGAME": "Int64", "FAQSHOP": "Int64",
            "FAQTOTAL": "Int64", "FAQTRAVL": "Int64",
            "KEYMED": str,
            "KEYMED_0": "Int64", "KEYMED_1": "Int64", "KEYMED_2": "Int64",
            "KEYMED_3": "Int64", "KEYMED_4": "Int64", "KEYMED_5": "Int64",
            "KEYMED_6": "Int64", "KEYMED_7": "Int64",
            "LDELTOTAL": "Int64", "LIMMTOTAL": "Int64", "MMSCORE": "Int64",
            "NXGAIT": "Int64", "PTCOGBEG": "Int64",
            "TOTAL13": float, "TOTSCORE": float, "TRAASCOR": "Int64",
            "PTEDUCAT": "Int64", "PTENGSPK": "Int64", "PTHOME": "Int64",
            "PTLANGTTL": "Int64", "PTMARRY": "Int64",
            "PTNLANG": str,
            "PTNLANG_1": "Int64", "PTNLANG_2": "Int64", "PTNLANG_3": "Int64",
            "PTNOTRT": "Int64", "PTPLANG": "Int64", "PTRTYR": "Int64",
            "PTSPOTTIM": float, "PTWORK": "Int64",
            "DIAGNOSIS": "Int64", "MRI_acquired": "Int64",
            "research_group": str, "GENOTYPE": str,
            "PTGENDER": "Int64", "AXDPMOOD": "Int64",
            "BCDEPRES": "Int64", "BCDPMOOD": "Int64",
            "DXDEP": "Int64", "DXNODEP": "Int64",
            "GDTOTAL": "Int64", "GLUCOSE": float,
            "HMHYPERT": "Int64", "HMT40": float,
            "NPID": "Int64", "NPID8": "Int64", "NPIDSEV": "Int64",
            "NPIDTOT": "Int64",
            "NPIK": "Int64", "NPIK1": "Int64", "NPIK2": "Int64",
            "NPIK3": "Int64", "NPIK4": "Int64", "NPIK5": "Int64",
            "NPIK6": "Int64", "NPIK7": "Int64", "NPIK8": "Int64",
            "NPIK9A": "Int64", "NPIK9B": "Int64", "NPIK9C": "Int64",
            "NPIKSEV": "Int64", "NPIKTOT": "Int64",
            "RCT1408": float, "RCT19": float, "RCT20": float,
            "VSBPDIA": float, "VSBPSYS": float,
            "VSHEIGHT": float, "VSHTUNIT": "Int64",
            "VSWEIGHT": float, "VSWTUNIT": "Int64",
            "EXABUSE": "Int64",
            "MH14AALCH": "Int64", "MH14ALCH": "Int64",
            "MH16ASMOK": float, "MH16CSMOK": float, "MH16SMOK": "Int64",
            "AXCHEST": "Int64", "AXFALL": "Int64",
            "AXHDACHE": "Int64", "AXINSOMN": "Int64",
            "AXMUSCLE": "Int64", "AXVISION": "Int64",
            "BCCHEST": "Int64", "BCFALL": "Int64",
            "BCHDACHE": "Int64", "BCINSOMN": "Int64",
            "BCMUSCLE": "Int64", "BCSTROKE": "Int64", "BCVISION": "Int64",
            "BSXCHRON": "Int64", "BSXSEVER": "Int64", "BSXSYMNO": "Int64",
            "HMSCORE": "Int64", "HMSOMATC": "Int64", "HMSTROKE": "Int64",
            "INCVISUAL": "Int64", "MH12RENA": "Int64", "MH4CARD": "Int64",
            "MHCUR": "Int64", "MHNUM": "Int64", "MHSTAB": "Int64",
            "NPPDXI": "Int64", "NPPDXJ": "Int64",
            "NXAUDITO": "Int64", "PXHEART": "Int64", "PXPERIPH": "Int64",
            "MHNUM_1": "Int64", "MHNUM_2": "Int64", "MHNUM_3": "Int64",
            "MHNUM_4": "Int64", "MHNUM_5": "Int64", "MHNUM_6": "Int64",
            "MHNUM_7": "Int64", "MHNUM_8": "Int64", "MHNUM_9": "Int64",
            "MHNUM_10": "Int64", "MHNUM_11": "Int64", "MHNUM_12": "Int64",
            "MHNUM_13": "Int64", "MHNUM_14": "Int64", "MHNUM_15": "Int64",
            "MHNUM_16": "Int64", "MHNUM_17": "Int64", "MHNUM_18": "Int64",
            "MHNUM_19": "Int64",
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def encode_row(value_str, unique_values: list) -> list:
        """One-hot encode a single pipe-delimited cell value."""
        one_hot = [0] * len(unique_values)
        if pd.isna(value_str) or value_str in ("nan", ""):
            return [np.nan] * len(unique_values)
        for part in str(value_str).split("|"):
            try:
                idx = unique_values.index(int(float(part)))
                one_hot[idx] = 1
            except (ValueError, IndexError):
                continue
        return one_hot

    def encode_var(self, var: str) -> pd.DataFrame:
        """One-hot encode a multi-hot column in ``self.data_df``."""
        series = self.data_df[var].astype(str)
        unique_values: set[int] = set()
        for val in series:
            if pd.isna(val) or val in ("nan", ""):
                continue
            for part in str(val).split("|"):
                try:
                    unique_values.add(int(float(part)))
                except ValueError:
                    pass

        unique_values_sorted = sorted(unique_values)
        log.info("(%s) Unique values: %s", var, unique_values_sorted)

        one_hot_cols = series.apply(
            lambda x: self.encode_row(x, unique_values_sorted)
        )
        return pd.DataFrame(
            one_hot_cols.tolist(),
            columns=[f"{var}_{v}" for v in unique_values_sorted],
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the original working directory.

        When running under Hydra the CWD is changed to the run output
        directory, so relative paths must be resolved against the
        original project root.
        """
        p = Path(path)
        if p.is_absolute():
            return p
        try:
            import hydra
            return Path(hydra.utils.get_original_cwd()) / p
        except (ImportError, AttributeError, ValueError):
            # Not running under Hydra — resolve against actual CWD
            return p.resolve()

    def load_data(self) -> None:
        """Load the main subjects table and the MRI table from CSV."""
        data_path = self._resolve_path(self.data_path)
        mri_path = self._resolve_path(self.mri_path)
        log.info("Loading data from %s and %s", data_path, mri_path)
        self.data_df = pd.read_csv(data_path)
        self.mri_df = pd.read_csv(mri_path)

    def encode_multihot_variables(self) -> None:
        """One-hot encode all multi-hot columns and concatenate them."""
        log.info("Encoding multi-hot variables: %s", self.MULTIHOT_VARS)
        encoded_dfs = [self.encode_var(var) for var in self.MULTIHOT_VARS]
        self.data_df = pd.concat([self.data_df] + encoded_dfs, axis=1)

    def coerce_numeric_columns(self) -> None:
        """Force-coerce string-contaminated numeric columns and clean CV."""
        log.info("Coercing numeric columns")
        for col in self.NUMERIC_COERCE_COLS:
            self.data_df.loc[:, col] = pd.to_numeric(
                self.data_df.loc[:, col], errors="coerce"
            )
        # CV column has trailing '%' characters
        self.data_df.loc[:, "CV"] = (
            pd.DataFrame(
                self.data_df.loc[:, "CV"].astype(str).str.split("%").to_list()
            )[0].astype(float)
        )

    def compute_bmi(self) -> None:
        """Compute HEIGHT (cm), WEIGHT (kg), and BMI."""
        log.info("Computing BMI")
        self.data_df["HEIGHT"] = self.data_df["VSHEIGHT"].where(
            self.data_df["VSHTUNIT"] == 2,
            self.data_df["VSHEIGHT"] * 2.54,
        )
        self.data_df["WEIGHT"] = self.data_df["VSWEIGHT"].where(
            self.data_df["VSWTUNIT"] == 2,
            self.data_df["VSWEIGHT"] * 0.453592,
        )
        self.data_df["BMI"] = self.data_df["WEIGHT"] / (
            (self.data_df["HEIGHT"] / 100) ** 2
        )

    def filter_cn_subjects(self) -> None:
        """Split CN subjects into those with multiple vs. single diagnoses."""
        log.info("Filtering CN subjects")
        data_cn_df = self.data_df[self.data_df["research_group"] == "CN"]
        subjects_cn_id = data_cn_df["subject_id"].unique()

        self._subjects_multiple_dx = []
        self._subjects_one_dx = []

        for subject in subjects_cn_id:
            n_dx = len(
                self.data_df[self.data_df["subject_id"] == subject][
                    "DIAGNOSIS"
                ]
                .dropna()
                .unique()
            )
            if n_dx > 1:
                self._subjects_multiple_dx.append(subject)
            elif n_dx == 1:
                self._subjects_one_dx.append(subject)

        self._data_cn_change_df = self.data_df[
            self.data_df["subject_id"].isin(self._subjects_multiple_dx)
        ]
        self._data_cn_no_change_df = self.data_df[
            self.data_df["subject_id"].isin(self._subjects_one_dx)
        ]
        log.info(
            "CN subjects — multiple diagnoses: %d, single diagnosis: %d",
            len(self._subjects_multiple_dx),
            len(self._subjects_one_dx),
        )

    def build_transition_cohort(self) -> None:
        """Build the cohort of CN subjects who later transitioned to MCI/AD."""
        log.info("Building transition cohort")
        tp = self.TIMEPOINTS_BASELINE
        tp_all = self.TIMEPOINTS_ALL
        filtered_data = []

        for subject_id in self._subjects_multiple_dx:
            subject_df = self._data_cn_change_df[
                self._data_cn_change_df["subject_id"] == subject_id
            ]
            subject_mri_df = self.mri_df[self.mri_df["PTID"] == subject_id]

            cond_baseline = subject_df["visit"].isin(tp)

            baseline_df = (
                subject_df[cond_baseline]
                .sort_values(by=["visit"], ascending=False)
                .infer_objects(copy=False)
                .bfill(axis=0)
                .ffill(axis=0)
                .iloc[-1:, :]
            )

            cond_all = subject_df["visit"].isin(tp_all)
            if len(subject_df[~cond_all]) > 1:
                months = (
                    subject_df[~cond_baseline]["visit"]
                    .str.split("m", expand=True)[1]
                    .dropna()
                    .astype(int)
                    .sort_values()
                )
                subject_df_months = subject_df.loc[months.index].copy()
                subject_df_months["months"] = months
                subject_df_months = subject_df_months.dropna(
                    subset=["DIAGNOSIS"]
                )

                if len(subject_df_months) > 0:
                    dx_cn_before_12 = (
                        subject_df_months[subject_df_months["months"] <= 12][
                            "DIAGNOSIS"
                        ]
                        == 1
                    ).all()
                    # At least one MCI/AD diagnosis after the first 12 months
                    dx_any_after_12 = (
                        subject_df_months[subject_df_months["months"] > 12][
                            "DIAGNOSIS"
                        ]
                        != 1
                    ).any()

                    if dx_cn_before_12 and dx_any_after_12:
                        baseline_df = baseline_df.copy()
                        baseline_df["study_duration"] = months.max()
                        baseline_df["last_diagnosis"] = subject_df_months.iloc[-1]["DIAGNOSIS"]
                        filtered_data.append(baseline_df)

        self.subjects_transition_df = pd.concat(filtered_data).astype(
            self.datatypes_dict
        )
        self.subjects_transition_df["transition"] = 1
        log.info(
            "Transition cohort: %d subjects",
            self.subjects_transition_df.shape[0],
        )

    def build_no_transition_cohort(self) -> None:
        """Build the cohort of CN subjects who remained CN throughout."""
        log.info("Building no-transition cohort")
        tp = self.TIMEPOINTS_BASELINE
        tp_all = self.TIMEPOINTS_ALL
        filtered_data = []

        for subject_id in self._subjects_one_dx:
            subject_df = self._data_cn_no_change_df[
                self._data_cn_no_change_df["subject_id"] == subject_id
            ]
            cond_baseline = subject_df["visit"].isin(tp)
            baseline_df = (
                subject_df[cond_baseline]
                .sort_values(by=["visit"], ascending=False)
                .infer_objects(copy=False)
                .bfill(axis=0)
                .ffill(axis=0)
                .iloc[-1:, :]
            )

            cond_all = subject_df["visit"].isin(tp_all)
            if len(subject_df[~cond_all]) > 1:
                months = (
                    subject_df[~cond_baseline]["visit"]
                    .str.split("m", expand=True)[1]
                    .dropna()
                    .astype(int)
                    .sort_values()
                )
                subject_df_months = subject_df.loc[months.index].copy()
                subject_df_months["months"] = months
                subject_df_months = subject_df_months.dropna(
                    subset=["DIAGNOSIS"]
                )

                if len(subject_df_months) > 0:
                    dx_last = subject_df_months.iloc[-1]["DIAGNOSIS"]
                    baseline_df = baseline_df.copy()
                    baseline_df["study_duration"] = months.max()
                    baseline_df["last_diagnosis"] = dx_last
                    filtered_data.append(baseline_df)

        self.subjects_no_transition_df = pd.concat(filtered_data).astype(
            self.datatypes_dict
        )
        self.subjects_no_transition_df["transition"] = 0
        log.info(
            "No-transition cohort: %d subjects",
            self.subjects_no_transition_df.shape[0],
        )

    def match_cohorts(self) -> None:
        """Match each transition subject to the closest no-transition subject."""
        log.info("Matching transition ↔ no-transition cohorts")
        selected_pairs = []
        pool_df = self.subjects_no_transition_df.copy()
        subjects_id_transition = self.subjects_transition_df["subject_id"]

        for subject in subjects_id_transition:
            num_vals = self.subjects_transition_df[
                self.subjects_transition_df["subject_id"] == subject
            ][self.NUM_MATCH_VARS]
            cat_vals = self.subjects_transition_df[
                self.subjects_transition_df["subject_id"] == subject
            ][self.CAT_MATCH_VARS]

            diff_df = (pool_df[self.NUM_MATCH_VARS] - num_vals.values).astype(
                float
            )
            diff_df["norm"] = np.linalg.norm(diff_df, axis=1)
            diff_df = diff_df.sort_values(by=["norm"], ascending=True)
            diff_df = pd.merge(
                left=diff_df,
                right=pool_df[self.CAT_MATCH_VARS],
                left_index=True,
                right_index=True,
            )
            cat_sim = np.all(
                diff_df[self.CAT_MATCH_VARS] == cat_vals.values, axis=1
            )
            candidates_df = diff_df.loc[cat_sim]
            best = pool_df.loc[[candidates_df.index[0]]]
            selected_pairs.append(best)
            pool_df = pool_df[
                pool_df["subject_id"] != best["subject_id"].item()
            ]

        self.subjects_pairs_df = pd.concat(selected_pairs)
        self.remaining_test_df = pool_df

        # Assign paired-group indices
        group = np.arange(1, self.subjects_transition_df.shape[0] + 1)
        self.subjects_transition_df["group"] = group
        self.subjects_pairs_df["group"] = group
        self.joint_dataset_df = pd.concat(
            [self.subjects_transition_df, self.subjects_pairs_df], axis=0
        )
        log.info(
            "Matched %d pairs; %d remaining unmatched controls",
            len(selected_pairs),
            self.remaining_test_df.shape[0],
        )

    def select_features(self) -> None:
        """Drop near-zero-variance features using ``VarianceThreshold``."""
        log.info(
            "Selecting features with variance > %s", self.variance_threshold
        )
        features = list(
            set(self.joint_dataset_df.columns) - set(self.CAT_VARS)
        )
        selector = VarianceThreshold(threshold=self.variance_threshold)
        selector.fit(self.joint_dataset_df[features])
        mask = selector.get_support()
        self.keep_features = list(np.array(features)[mask])
        self.keep_features.extend(
            [
                "subject_id", "visit", "group", "transition",
                "GENOTYPE", "research_group", "subject_date",
            ]
        )
        log.info("Keeping %d features", len(self.keep_features))

        self.joint_dataset_df = self.joint_dataset_df[self.keep_features]
        remaining_cols = [c for c in self.keep_features if c != "group"]
        self.remaining_test_df = self.remaining_test_df[remaining_cols]

    def export_datasets(self) -> None:
        """Write the final joint and remaining-test datasets to CSV."""
        out_dir = self._resolve_path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joint_path = out_dir / "joint_dataset.csv"
        remaining_path = out_dir / "remaining_test.csv"
        mri_path = out_dir / "mri_joint_dataset.csv"

        log.info("Exporting joint dataset → %s", joint_path)
        self.joint_dataset_df.to_csv(joint_path)
        log.info("Shape of joint dataset: %s", self.joint_dataset_df.shape)

        log.info("Exporting remaining test set → %s", remaining_path)
        self.remaining_test_df.to_csv(remaining_path)
        log.info("Shape of remaining test set: %s", self.remaining_test_df.shape)

        # MRI subset
        mri_joint_idx = list(
            set(self.joint_dataset_df["subject_id"]).intersection(
                set(self.mri_df["PTID"])
            )
        )
        mri_joint_df = self.mri_df.set_index("PTID").loc[mri_joint_idx]
        mri_joint_df = mri_joint_df[
            (mri_joint_df["VISCODE2"] == "sc")
            | (mri_joint_df["VISCODE2"] == "scmri")
        ]
        mri_joint_df = mri_joint_df.sort_values(by="EXAMDATE")
        mri_joint_df = mri_joint_df[
            ~mri_joint_df.index.duplicated(keep="first")
        ]

        log.info("Exporting MRI joint dataset → %s", mri_path)
        mri_joint_df.to_csv(mri_path)
        log.info("Shape of MRI joint dataset: %s", mri_joint_df.shape)

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full preprocessing pipeline."""
        log.info("Starting ADNI preprocessing pipeline")
        self.load_data()
        self.encode_multihot_variables()
        self.coerce_numeric_columns()
        self.compute_bmi()
        self.filter_cn_subjects()
        self.build_transition_cohort()
        self.build_no_transition_cohort()
        self.match_cohorts()
        self.select_features()
        self.export_datasets()
        log.info("ADNI preprocessing complete")
