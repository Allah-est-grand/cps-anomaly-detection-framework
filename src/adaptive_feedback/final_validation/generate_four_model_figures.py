"""Recreate seven publication-ready figures from saved four-model experiment results.

No model imports, dataset loading, or model training.
Run: python generate_four_model_figures.py --csv path/to/multi_seed_four_model_results.csv
Default outputs: sibling directory of the input CSV, called four_model_final_figures.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = [42, 123, 2024, 2025, 3407]
MODELS = ['XGBoost', 'RandomForest', 'LinearSVM', 'LSTM']
STAGES = ['baseline', 'weighted_retraining']
DISPLAY = {'XGBoost': 'XGBoost', 'RandomForest': 'Random Forest', 'LinearSVM': 'Linear SVM', 'LSTM': 'LSTM'}
STAGE_NAMES = {'baseline': 'Baseline', 'weighted_retraining': 'Weighted retraining'}
METRICS = [
    ('f1_score', 'F1 score', 'figure_01_f1'),
    ('recall', 'Recall', 'figure_02_recall'),
    ('precision', 'Precision', 'figure_03_precision'),
    ('false_negative', 'False negatives (FN)', 'figure_04_false_negatives'),
    ('false_positive', 'False positives (FP)', 'figure_05_false_positives'),
]


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {'seed','model_name','model_type','evaluation_n','accuracy',
                'precision','recall','f1_score','false_positive','false_negative','status'}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if df.shape[0] != 40 or not df.status.eq('success').all():
        raise ValueError('Expected exactly 40 successful records (4 models × 2 stages × 5 seeds).')
    keys = df[['seed','model_name','model_type']].apply(tuple,axis=1)
    required = {(s,m,t) for s in SEEDS for m in MODELS for t in STAGES}
    if len(set(keys)) != 40 or set(keys) != required:
        raise ValueError('Missing, duplicated or unexpected seed/model/stage records.')
    numeric = ['evaluation_n','accuracy','precision','recall','f1_score', 'false_positive','false_negative']
    df[numeric] = df[numeric].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite recorded metric or count.')
    if not df[['accuracy','precision','recall','f1_score']].ge(0).all().all() or not df[['accuracy','precision','recall','f1_score']].le(1).all().all():
        raise ValueError('Metric outside [0,1].')
    if (df[['evaluation_n','false_positive','false_negative']] < 0).any().any():
        raise ValueError('Negative counts.')
    if not (df.loc[df.model_name!='LSTM','evaluation_n']==288344).all() or not (df.loc[df.model_name=='LSTM','evaluation_n']==288335).all():
        raise ValueError('Unexpected evaluation population; inspect split/window protocol.')
    return df


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out/f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out/f'{name}.svg', bbox_inches='tight')
    plt.close(fig)


def grouped_model_bars(df: pd.DataFrame, metric: str, ylabel: str, name: str, out: Path):
    data = df.groupby(['model_name','model_type'])[metric].agg(['mean','std'])
    fig, ax = plt.subplots(figsize=(10.5,6.2))
    x = np.arange(len(MODELS)); width=0.37
    for idx, stage in enumerate(STAGES):
        means = [data.loc[(m,stage),'mean'] for m in MODELS]
        stds = [data.loc[(m,stage),'std'] for m in MODELS]
        ax.bar(x+(idx-.5)*width, means, width, yerr=stds, capsize=4,
               label=STAGE_NAMES[stage], alpha=.85)
    ax.set_xticks(x, [DISPLAY[m] for m in MODELS]); ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} across five random seeds (mean ± sample SD)')
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True); ax.legend()
    if metric in ('f1_score','recall','precision'):
        ax.set_ylim(0,1.04)
    else:
        ax.text(.01,-.17,'Counts are within each model’s evaluation population; LSTM excludes 9 test endpoints.',
                transform=ax.transAxes,fontsize=8)
    save(fig,out,name)


def stability_plot(df: pd.DataFrame, out: Path):
    fig, ax=plt.subplots(figsize=(11,6.5))
    for model in MODELS:
        for stage in STAGES:
            part=df[(df.model_name==model)&(df.model_type==stage)].set_index('seed').loc[SEEDS]
            ax.plot(range(5),part.f1_score.to_numpy(),marker='o' if stage=='baseline' else 's',
                    linestyle='-' if stage=='baseline' else '--',
                    label=f'{DISPLAY[model]} — {STAGE_NAMES[stage]}')
    ax.set_xticks(range(5),[str(s) for s in SEEDS]);ax.set_xlabel('Seed');ax.set_ylabel('F1 score')
    ax.set_title('F1 stability: paired baseline and retraining per seed')
    ax.grid(alpha=.23);ax.legend(ncol=2,fontsize=8,loc='best')
    save(fig,out,'figure_06_f1_seed_stability')


def paired_table(df: pd.DataFrame)->pd.DataFrame:
    wide=df.pivot(index=['seed','model_name'],columns='model_type',values=[x[0] for x in METRICS])
    records=[]
    for model in MODELS:
        for metric,label,_ in METRICS:
            d=(wide.loc[(slice(None),model),(metric,'weighted_retraining')].to_numpy()
               -wide.loc[(slice(None),model),(metric,'baseline')].to_numpy())
            records.append(dict(model=model,metric=metric,comparison='weighted_retraining_minus_baseline',
                        n_pairs=len(d),mean_difference=float(d.mean()),sample_sd=float(d.std(ddof=1)),
                        **{f'seed_{seed}':float(val) for seed,val in zip(SEEDS,d)}))
    return pd.DataFrame.from_records(records)


def paired_plot(paired: pd.DataFrame,out:Path):
    # Show within-model changes in two comparable rate metrics, not raw FNs/FPs on different axes.
    fig,ax=plt.subplots(figsize=(10.5,6.2)); x=np.arange(4); width=.38
    for j,metric in enumerate(['f1_score','recall']):
        part=paired[paired.metric==metric].set_index('model').loc[MODELS]
        ax.bar(x+(j-.5)*width,part.mean_difference.to_numpy(),width,
               yerr=part.sample_sd.to_numpy(),capsize=4,label=f'Δ{metric.replace("_score", "").title()}')
    ax.axhline(0,color='black',linewidth=.8)
    ax.set_xticks(x,[DISPLAY[m] for m in MODELS]); ax.set_ylabel('Weighted − baseline (absolute metric units)')
    ax.set_title('Paired F1 and recall changes across five seeds (mean ± sample SD)')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,out,'figure_07_paired_changes')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True,help='Path to existing multi_seed_four_model_results.csv')
    p.add_argument('--output',type=Path,help='Directory for figures; defaults beside CSV')
    args=p.parse_args()
    csv=args.csv.expanduser().resolve(); out=(args.output or csv.parent/'four_model_final_figures').expanduser().resolve()
    df=load_and_validate(csv); out.mkdir(parents=True,exist_ok=True)
    for metric,label,name in METRICS: grouped_model_bars(df,metric,label,name,out)
    stability_plot(df,out)
    paired=paired_table(df);paired.to_csv(out/'paired_statistics_from_csv.csv',index=False)
    paired_plot(paired,out)
    readme=('Generated only from the supplied 40-row results CSV. No training.\n'
            '7 figures each in PNG 300 dpi and editable SVG, plus paired CSV.\n'
            'Five predeclared seeds; mean ± sample SD (ddof=1).\n'
            'Classical models: 288344 rows; LSTM: 288335 valid 10-row window endpoints.\n'
            'These results describe weighted retraining, NOT causal validation of adaptive anomaly generation.\n')
    (out/'README.txt').write_text(readme,encoding='utf-8')
    print('PASS: 40 unique successful records, expected seeds and evaluation populations')
    print('PASS: 7 PNG figures, 7 SVG figures, paired CSV, README')
    print(f'OUTPUT: {out}')

if __name__=='__main__':main()
