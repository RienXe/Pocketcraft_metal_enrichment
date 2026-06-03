"""Module for evaluating metal binding prediction performance."""

import os
import glob
import json
import argparse
from tqdm import tqdm
from utils.metrics import process_single_pair
from utils.plotting import plot_roc_curve, plot_per_metal_histograms, plot_per_metal_category_histograms

TARGET_METALS = ['ZN', 'CA', 'MG', 'FE', 'MN', 'CU', 'NI', 'CO', 'NA', 'K']

def main(gt_dir, pred_dir, out_dir):
    """
    Run evaluation comparing ground truth structures against predicted structures.

    This function calculates exact and site-level matching metrics,
    identifies false positives and false negatives, outputs a comprehensive JSON
    report, and generates ROC curves and histograms.

    Args:
        gt_dir (str): Directory containing the ground truth CIF/PDB files.
        pred_dir (str): Directory containing the predicted CIF files.
        out_dir (str): Output directory where metrics and plots will be saved.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
    global_site, global_exact, global_top1 = [], [], []
    total_metrics = {'gt_all_metals': 0, 'gt_valid_metals': 0, 'gt_functional_metals': 0, 'gt_possible_metals': 0, 'gt_negative_metals': 0, 'total_predicted_metals': 0}
    gt_files = glob.glob(os.path.join(gt_dir, "*.cif")) + glob.glob(os.path.join(gt_dir, "*.pdb"))

    for gt_path in tqdm(gt_files, desc="Batch Evaluation"):
            base_name = os.path.splitext(os.path.basename(gt_path))[0]
            pred_path = os.path.join(pred_dir, f"{base_name}_predicted.cif")
            
            try:
                site_r, exact_r, top1_r, counts = process_single_pair(gt_path, pred_path, TARGET_METALS)
                if counts:
                    total_metrics['gt_all_metals'] += counts.get('gt_all', 0)
                    total_metrics['gt_valid_metals'] += counts.get('gt_valid', 0)
                    total_metrics['gt_functional_metals'] += counts.get('gt_functional', 0)
                    total_metrics['gt_possible_metals'] += counts.get('gt_possible', 0)
                    total_metrics['gt_negative_metals'] += counts.get('gt_negative', 0)
                    total_metrics['total_predicted_metals'] += counts.get('pred_all', 0)
                    
                if site_r is not None:
                    global_site.extend(site_r)
                    global_exact.extend(exact_r)
                    global_top1.extend(top1_r)
            except Exception as e:
                print(f"\n[Warning] Failed evaluating {base_name}: {e}")

    per_metal_stats = {}
    for m in TARGET_METALS:
        exact_match_functional = sum(1 for p in global_exact if p['gt_type'] == m and p['status'] == 'TP' and p['cat'] == 'functional')
        site_match_functional = sum(1 for p in global_site if p['gt_type'] == m and p['status'] == 'TP' and p['cat'] == 'functional')
        
        exact_match_possible = sum(1 for p in global_exact if p['gt_type'] == m and p['status'] == 'TP' and p['cat'] == 'possible')
        site_match_possible = sum(1 for p in global_site if p['gt_type'] == m and p['status'] == 'TP' and p['cat'] == 'possible')

        per_metal_stats[m] = {
            "GT_Functional_Total": sum(1 for p in global_top1 if p['gt_type'] == m and p['cat'] == 'functional'),
            "GT_Possible_Total": sum(1 for p in global_top1 if p['gt_type'] == m and p['cat'] == 'possible'),
            "GT_Valid_Total": sum(1 for p in global_top1 if p['gt_type'] == m and p['cat'] in ['functional', 'possible']),
            
            "Exact_Matches_Functional": exact_match_functional,
            "Exact_Matches_Possible": exact_match_possible,
            "Site_Matches_Functional": site_match_functional,
            "Site_Matches_Possible": site_match_possible,

            "Total_Exact_Matches": exact_match_functional + exact_match_possible,
            "Total_Site_Matches": site_match_functional + site_match_possible,

            "FN_Missed_Completely": {
                "In_functional_Site": {
                    "count": sum(1 for p in global_exact if p['gt_type'] == m and p['status'] == 'FN_Missed' and p['cat'] == 'functional'),
                    "pdb_ids": sorted(list(set([p['pdb_id'] for p in global_exact if p['gt_type'] == m and p['status'] == 'FN_Missed' and p['cat'] == 'functional'])))
                },
                "In_possible_Site": {
                    "count": sum(1 for p in global_exact if p['gt_type'] == m and p['status'] == 'FN_Missed' and p['cat'] == 'possible'),
                    "pdb_ids": sorted(list(set([p['pdb_id'] for p in global_exact if p['gt_type'] == m and p['status'] == 'FN_Missed' and p['cat'] == 'possible'])))
                }
            },

            "FP_True_Background_Hallucinations": {
                "Hallucinated_as_functional": {
                    "count": sum(1 for p in global_exact if p['pred_type'] == m and p['status'] == 'FP_Background' and p['cat'] == 'functional'),
                    "pdb_ids": sorted(list(set([p['pdb_id'] for p in global_exact if p['pred_type'] == m and p['status'] == 'FP_Background' and p['cat'] == 'functional'])))
                },
                "Hallucinated_as_possible": {
                    "count": sum(1 for p in global_exact if p['pred_type'] == m and p['status'] == 'FP_Background' and p['cat'] == 'possible'),
                    "pdb_ids": sorted(list(set([p['pdb_id'] for p in global_exact if p['pred_type'] == m and p['status'] == 'FP_Background' and p['cat'] == 'possible'])))
                },
                "Hallucinated_in_Empty_Space": {
                    "count": sum(1 for p in global_exact if p['pred_type'] == m and p['status'] == 'FP_Background' and p['cat'] == 'Negative'),
                    "pdb_ids": sorted(list(set([p['pdb_id'] for p in global_exact if p['pred_type'] == m and p['status'] == 'FP_Background' and p['cat'] == 'Negative'])))
                }
            },
            
            "FP_Correct_Site_Wrong_Metal": {
                "count": sum(1 for p in global_exact if p['pred_type'] == m and p['status'] == 'FN_WrongType'),
                "pdb_ids": sorted(list(set([p['pdb_id'] for p in global_exact if p['pred_type'] == m and p['status'] == 'FN_WrongType'])))
            }
        }

    report = {
        "1_Dataset_Ground_Truth_Counts": total_metrics,
        "2_Site_Detection_Matches": {
            "Overall_Valid_Sites_Found": sum(1 for p in global_site if p['y_true'] == 1 and p['status'] == "TP"),
            "Functional_Sites_Found": sum(1 for p in global_site if p['cat'] == 'functional' and p['status'] == "TP"),
            "Possible_Sites_Found": sum(1 for p in global_site if p['cat'] == 'possible' and p['status'] == "TP")
        },
        "5_False_Positives_Counts_Summary": {
            "Predicted_on_Negative_GT_Artifacts": sum(1 for p in global_site if p['status'] == 'FP_Negative'),
            "Correct_Site_but_Wrong_Metal_Type": sum(1 for p in global_exact if p['status'] == 'FN_WrongType'),
            "True_Background_Hallucination_(False_Site)": sum(1 for p in global_site if p['status'] == 'FP_Background')
        },
        "6_Per_Metal_Type_Detailed_Analysis": per_metal_stats
    }
    
    json_path = os.path.join(out_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Saved metrics to {json_path}")

    print("\n--- Generating Plots ---")
    
    plot_roc_curve({
        "Overall Site Detection": [p for p in global_site if p['cat'] in ['functional', 'possible', 'Negative', 'Background']],
        "Functional (>=3 residues) vs Neg/Bg": [p for p in global_site if p['cat'] in ['functional', 'Negative', 'Background']],
        "Possible (<3 residues) vs Neg/Bg": [p for p in global_site if p['cat'] in ['possible', 'Negative', 'Background']]
    }, "ROC: Metal Site Detection", os.path.join(out_dir, "ROC_Site_Detection.png"))

    plot_roc_curve({
        "Overall Exact Match": [p for p in global_exact if p['cat'] in ['functional', 'possible', 'Negative', 'Background']],
        "Functional (>=3 residues) vs Neg/Bg": [p for p in global_exact if p['cat'] in ['functional', 'Negative', 'Background']],
        "Possible (<3 residues) vs Neg/Bg": [p for p in global_exact if p['cat'] in ['possible', 'Negative', 'Background']]
    }, "ROC: Exact Metal Match", os.path.join(out_dir, "ROC_Exact_Match.png"))

    plot_roc_curve({
        "Overall Top-1 Match": [p for p in global_top1 if p['cat'] in ['functional', 'possible', 'Negative', 'Background']],
        "Functional (>=3 residues) vs Neg/Bg": [p for p in global_top1 if p['cat'] in ['functional', 'Negative', 'Background']],
        "Possible (<3 residues) vs Neg/Bg": [p for p in global_top1 if p['cat'] in ['possible', 'Negative', 'Background']]
    }, "ROC: Top-1 Exact Match", os.path.join(out_dir, "ROC_Top1_Match.png"))

    metal_dict = {}
    for m in TARGET_METALS:
        metal_data = [p for p in global_exact if (p['gt_type'] == m and p['cat'] in ['functional', 'possible']) or (p['pred_type'] == m and p['y_true'] == 0)]
        if len(metal_data) > 0:
            metal_dict[f"Metal: {m}"] = metal_data
            
    plot_roc_curve(
        metal_dict, 
        "ROC: Exact Match by Target Element", 
        os.path.join(out_dir, "ROC_Per_Metal.png")
    )

    print("Generating Functional Per-Metal ROCs...")
    functional_metal_roc = {}
    for m in TARGET_METALS:
        metal_data = [
            p for p in global_exact 
            if (p['gt_type'] == m and p['cat'] == 'functional') or 
               (p['pred_type'] == m and p['y_true'] == 0)
        ]
        if len(metal_data) > 0:
            functional_metal_roc[f"{m} (Functional)"] = metal_data
            
    plot_roc_curve(
        functional_metal_roc, 
        "ROC: Exact Match - Functional Sites Only", 
        os.path.join(out_dir, "ROC_Per_Metal_Functional.png")
    )

    print("Generating Top-1 Functional ROC...")
    plot_roc_curve({
        "Top-1 Overall": [p for p in global_top1 if p['cat'] in ['functional', 'possible', 'Negative', 'Background']],
        "Top-1 Functional Only": [p for p in global_top1 if p['cat'] in ['functional', 'Negative', 'Background']]
    }, "ROC: Top-1 Match (Quality Comparison)", os.path.join(out_dir, "ROC_Top1_Functional.png"))

    print("Plotting Per-Metal Histograms...")
    plot_per_metal_histograms(
        global_exact, TARGET_METALS, 'dist', 
        "Distance Error per Metal (Exact Matches)", 
        "Distance to Ground Truth (Å)", 
        os.path.join(out_dir, "Hist_Distances_Per_Metal.png"), 
        x_range=(0, 2.0)
    )

    plot_per_metal_histograms(
        global_exact, TARGET_METALS, 'y_score', 
        "Prediction Probability per Metal (Exact Matches)", 
        "Predicted Probability", 
        os.path.join(out_dir, "Hist_Prob_Exact_Per_Metal.png"), 
        x_range=(0, 1.0)
    )

    print("Plotting Functional Probability Histograms per metal...")
    plot_per_metal_category_histograms(
        global_exact, TARGET_METALS, 'y_score', 'functional',
        "Prediction Probability per Metal (Functional Exact Matches)", 
        "Predicted Probability", 
        os.path.join(out_dir, "Hist_Prob_Exact_Functional_Per_Metal.png"), 
        x_range=(0, 1.0)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted metal binding sites against ground truth structures.")
    parser.add_argument("--gt_dir", required=True, help="Directory containing the ground truth structure files (.pdb or .cif).")
    parser.add_argument("--pred_dir", required=True, help="Directory containing the predicted structure files (.cif).")
    parser.add_argument("--out_dir", required=True, help="Directory where evaluation metrics and plots will be saved.")
    args = parser.parse_args()
    main(args.gt_dir, args.pred_dir, args.out_dir)
