"""
Experiment: does adding more synthetic training examples for the rarest,
worst grades (G-Permanent, H-Near Death, I-Death) improve the model on
those grades, without hurting anything else?

Train has only 71 G, 130 H, 158 I rows out of 50,091 -- the model barely
sees these often enough to learn what distinguishes them from softer grades.
This script hand-writes additional synthetic reports across varied clinical
scenarios (not just paraphrases of one archetype) for these three grades,
appends them ONLY to the training split, retrains the exact same
architecture as severity_model.py, and evaluates on the SAME frozen
validation/test sets -- nothing about the eval data changes, so any
improvement (or lack of one) is a fair read.

Caveat stated up front, same rigor bar as the rest of this project: these
narratives are hand-written by one person (me) in one sitting. They add
variety in clinical scenario (medication overdose, wrong-site surgery,
missed sepsis, falls, transfusion reaction, restraint injury, anesthesia
complication, equipment failure, hospital-acquired infection, elopement)
that the original ~1,258-archetype generator may not have covered as
thoroughly, but they carry their own single-author bias (vocabulary,
sentence rhythm) that real reports from many different clinicians wouldn't
have. Treat any gain here as "worth trying at scale with a real generator
or real data," not as proof the technique works in production.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, cohen_kappa_score, mean_absolute_error, accuracy_score

from data_pipeline import train_val_test, FULL_LABELS, FULL_LABEL2ID, BUCKET_GROUPS, HURT_GRADES

UNITS = ["4 East Med-Surg", "ICU", "6 West Ortho", "L&D", "PACU", "3 South Oncology",
         "ED", "7 North Cardiac", "SICU", "Rehab Unit"]
SERVICES = ["Internal Medicine", "Orthopaedic Surgery", "Cardiology", "General Surgery",
            "Obstetrics", "Emergency Medicine", "Oncology", "Neurosurgery", "Anesthesiology", "Pediatrics"]
AGES = [4, 17, 29, 41, 53, 62, 68, 74, 81, 88]

# Each entry: (grade, scenario_key, narrative_template). Narrative language
# is written to actually carry the outcome (unlike the two real test-set
# misses, which read as caught near-misses) -- these are meant to give the
# model clear signal, on purpose, for this test.
SCENARIOS = [
    ("I-Harm-Death", "med_overdose",
     "Patient received a 10x overdose of {drug} due to a misplaced decimal "
     "point on the order. Cardiac arrest followed within the hour; code "
     "blue called, resuscitation unsuccessful. Patient was pronounced dead "
     "at {time}. Pharmacy and nursing leadership notified immediately."),
    ("I-Harm-Death", "wrong_site_surgery",
     "Surgical team began the procedure on the wrong side before the error "
     "was caught mid-operation. Patient suffered a fatal hemorrhage during "
     "the resulting emergency correction and could not be revived despite "
     "prolonged resuscitation efforts in the OR."),
    ("I-Harm-Death", "missed_sepsis",
     "Patient's worsening vital signs and lab trends consistent with sepsis "
     "went unaddressed for over 12 hours. By the time blood cultures "
     "returned, the patient had progressed to septic shock and multi-organ "
     "failure, and died the following morning despite ICU transfer."),
    ("I-Harm-Death", "transfusion_reaction",
     "Patient received a unit of blood that had not been properly "
     "cross-matched. Within minutes of the transfusion starting, the "
     "patient developed acute hemolytic shock and went into cardiac arrest. "
     "Resuscitation was unsuccessful; time of death was called by the "
     "attending."),
    ("I-Harm-Death", "restraint_asphyxiation",
     "Patient was placed in physical restraints for agitation without "
     "continuous monitoring as policy requires. Staff found the patient "
     "unresponsive and not breathing approximately 40 minutes later. Code "
     "blue was called; the patient could not be resuscitated."),

    ("H-Harm-Near Death", "anesthesia_complication",
     "During induction of anesthesia, the patient experienced a severe "
     "airway complication that was not recognized for several minutes. "
     "Oxygen saturation dropped critically low before the airway was "
     "secured. Patient was transferred to the ICU in critical condition "
     "and remains on a ventilator; the care team considers this a "
     "near-death event."),
    ("H-Harm-Near Death", "equipment_failure",
     "A ventilator alarm was silenced and not reset after a prior false "
     "alarm. The patient subsequently experienced a real disconnection "
     "event that went unnoticed for several minutes, resulting in severe "
     "hypoxia. The patient was resuscitated and stabilized in the ICU but "
     "suffered a cardiac arrest that required advanced life support."),
    ("H-Harm-Near Death", "med_overdose_survived",
     "Patient was given a dose of {drug} roughly eight times the ordered "
     "amount. The error was caught after the patient became unresponsive "
     "and bradycardic. Emergency reversal agents were administered and the "
     "patient was transferred to the ICU in critical but stable condition "
     "after a prolonged resuscitation."),
    ("H-Harm-Near Death", "post_op_hemorrhage",
     "Patient developed a large, unrecognized post-operative hemorrhage "
     "overnight. By morning rounds the patient was in hypovolemic shock "
     "and required emergent return to the OR and multiple blood "
     "transfusions to stabilize. The patient survived but spent a week in "
     "the ICU in critical condition."),
    ("H-Harm-Near Death", "infant_delivery_complication",
     "During delivery, a delay in recognizing fetal distress led to a "
     "prolonged period of oxygen deprivation before an emergency cesarean "
     "was performed. The infant required immediate resuscitation and "
     "therapeutic hypothermia; the care team classified this as a "
     "near-death event for the newborn."),

    ("G-Harm-Permanent", "wrong_med_neuro_injury",
     "Patient was given {drug} instead of the ordered medication due to a "
     "look-alike packaging mix-up. The resulting adverse reaction caused "
     "a stroke-like event with lasting weakness on one side. Neurology "
     "confirmed permanent deficit expected on follow-up imaging."),
    ("G-Harm-Permanent", "pressure_injury_untreated",
     "A high-risk pressure injury was not repositioned or monitored per "
     "protocol for an extended period. The wound progressed to a "
     "stage 4 injury with exposed bone before it was escalated. Wound "
     "care specialists confirmed the resulting tissue damage is permanent."),
    ("G-Harm-Permanent", "delayed_diagnosis_amputation",
     "Signs of a developing limb-threatening vascular complication were "
     "not escalated for several days. By the time surgery was performed, "
     "the tissue damage was irreversible and a partial amputation was "
     "required."),
    ("G-Harm-Permanent", "medication_hearing_loss",
     "Patient received an extended course of {drug} at a dose known to "
     "risk ototoxicity without the required monitoring labs being drawn. "
     "Audiology confirmed permanent bilateral hearing loss as a result."),
    ("G-Harm-Permanent", "fall_head_injury",
     "Patient fell from bed after a bed alarm was found disconnected "
     "during rounds. The fall resulted in a traumatic brain injury; "
     "neurology has confirmed the resulting cognitive deficits are "
     "expected to be permanent."),
]

DRUGS = ["insulin", "heparin", "morphine", "warfarin", "vancomycin", "potassium chloride"]


def make_augmented_rows(n_per_scenario=4, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for i, (grade, key, template) in enumerate(SCENARIOS):
        for j in range(n_per_scenario):
            unit = rng.choice(UNITS)
            service = rng.choice(SERVICES)
            age = int(rng.choice(AGES))
            drug = rng.choice(DRUGS)
            time = f"{rng.integers(0, 24):02d}:{rng.choice([0, 15, 30, 45]):02d}"
            narrative = template.format(drug=drug, time=time)
            prefix = f"Unit: {unit}. Service: {service}. Age: {age}."
            text = f"{prefix}\n{narrative}"
            rows.append({
                "event_no": f"SYNAUG-TRAIN-{key}-{j:03d}",
                "split": "train",
                "grade": grade,
                "full_label": FULL_LABEL2ID[grade],
                "hurt_label": 1,
                "text": text,
            })
    return pd.DataFrame(rows)


def bucket_probs_sum(proba, classes_, group_ids):
    cols = [i for i, c in enumerate(classes_) if c in group_ids]
    return proba[:, cols].sum(axis=1)


def evaluate(name, vectorizer, clf, val_df, test_df):
    X_test = vectorizer.transform(test_df["text"])
    proba = clf.predict_proba(X_test)
    preds = clf.classes_[np.argmax(proba, axis=1)]
    y_true = test_df["full_label"].to_numpy()

    acc = accuracy_score(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    kappa = cohen_kappa_score(y_true, preds, weights="quadratic")

    print(f"\n=== {name} ===")
    print(f"Exact accuracy: {acc:.4f}  MAE: {mae:.4f}  Quadratic kappa: {kappa:.4f}")

    present = sorted(set(y_true) | set(preds))
    report = classification_report(
        y_true, preds, labels=present,
        target_names=[FULL_LABELS[i] for i in present],
        output_dict=True, zero_division=0,
    )
    for g in ["G-Harm-Permanent", "H-Harm-Near Death", "I-Harm-Death"]:
        if g in report:
            r = report[g]
            print(f"  {g:20s} precision={r['precision']:.3f} recall={r['recall']:.3f} "
                  f"support={r['support']:.0f}")
        else:
            print(f"  {g:20s} not predicted / not present in this test set")

    serious_ids = set(BUCKET_GROUPS["serious"])
    triage = bucket_probs_sum(proba, clf.classes_, set(BUCKET_GROUPS["some"]) | serious_ids)
    is_serious = test_df["full_label"].isin(serious_ids).to_numpy()
    cutoff = 0.23611363921600065  # frozen production cutoff, unchanged
    flagged = triage >= cutoff
    missed = is_serious & ~flagged
    print(f"  Missed serious cases at production cutoff (0.236): {missed.sum()} / {is_serious.sum()}")
    if missed.sum():
        print("   ", test_df.loc[missed, "event_no"].tolist())
    return {"accuracy": acc, "mae": mae, "kappa": kappa, "missed_serious": int(missed.sum())}


def main():
    train_df, val_df, test_df = train_val_test()
    print(f"Baseline train size: {len(train_df)}")

    aug_df = make_augmented_rows()
    print(f"Augmented rows added (train only): {len(aug_df)}")
    print(aug_df["grade"].value_counts())

    # --- Baseline (exact severity_model.py recipe) ---
    vec_base = TfidfVectorizer(
        stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3,
        token_pattern=r"(?u)\b[\w-]+\b",
    )
    X_train_base = vec_base.fit_transform(train_df["text"])
    clf_base = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf_base.fit(X_train_base, train_df["full_label"])
    base_result = evaluate("BASELINE (no augmentation)", vec_base, clf_base, val_df, test_df)

    # --- Augmented ---
    train_aug = pd.concat([train_df, aug_df[["event_no", "split", "grade", "full_label", "hurt_label", "text"]]],
                           ignore_index=True)
    vec_aug = TfidfVectorizer(
        stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=3,
        token_pattern=r"(?u)\b[\w-]+\b",
    )
    X_train_aug = vec_aug.fit_transform(train_aug["text"])
    clf_aug = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    clf_aug.fit(X_train_aug, train_aug["full_label"])
    aug_result = evaluate(f"AUGMENTED (+{len(aug_df)} synthetic G/H/I rows)", vec_aug, clf_aug, val_df, test_df)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20}{'Baseline':>12}{'Augmented':>12}{'Delta':>12}")
    for k in ["accuracy", "mae", "kappa"]:
        print(f"{k:<20}{base_result[k]:>12.4f}{aug_result[k]:>12.4f}{aug_result[k]-base_result[k]:>+12.4f}")
    print(f"{'missed_serious':<20}{base_result['missed_serious']:>12d}{aug_result['missed_serious']:>12d}"
          f"{aug_result['missed_serious']-base_result['missed_serious']:>+12d}")


if __name__ == "__main__":
    main()
