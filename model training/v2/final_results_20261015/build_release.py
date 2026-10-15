"""Publish the verified run without retraining or altering its frozen evidence.

Usage: python build_release.py --run /path/to/20261015_rolling_personal
       --beamer /path/to/Bad_Blood_V2_Models_Explained.tex
"""
import argparse
import collections
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
OLD = V2/'final_results_20261013'
METHODS = ['population','personal_mean','bias','pair_fixed','pair_personal_hpo',
           'rolling_head','rolling_tail','rolling_selected']
NAMES = ['Population (no calibration)','Personal mean','Bias correction','Fixed PairSE',
         'Single-period selection','Head-only adaptation','Tail adaptation (adopted)','Scope selection (comparison)']
METRICS = ['mae','rmse','bias','error_sd','within_5','within_10','within_15']
LABELS = ['MAE','RMSE','Signed Bias','Sample error SD',r'$\leq5$ (\%)',r'$\leq10$ (\%)',r'$\leq15$ (\%)']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s)
def copy(a,b):
    b.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(a,b)
def fmt(x): return ' / '.join(f'{v:.2f}' for v in x)
def frame(title, body): return '\\begin{frame}{'+title+'}\n'+body+'\n\\end{frame}\n'


def metric_table(result, methods, mode='macro', size=r'\smalltable', rowlabels=None):
    rows = [size, r'\begin{tabular}{l'+('r'*len(methods))+r'}\toprule',
            'Metric & '+' & '.join(rowlabels or methods)+r'\\\midrule']
    for metric,label in zip(METRICS,LABELS):
        a=np.array([result[m][mode][metric] for m in methods])
        rank=np.abs(a) if metric=='bias' else -a if metric.startswith('within') else a
        best=rank.min(0)
        cells=[]
        for i in range(len(methods)):
            cells.append(' / '.join((r'\textbf{'+f'{a[i,d]*(100 if metric.startswith("within") else 1):.2f}'+'}')
                if abs(rank[i,d]-best[d])<1e-10 else f'{a[i,d]*(100 if metric.startswith("within") else 1):.2f}' for d in range(2)))
        rows.append(label+' & '+' & '.join(cells)+r'\\')
    return '\n'.join(rows+[r'\bottomrule\end{tabular}'])


def publish(run, beamer):
    complete=json.loads((run/'complete.json').read_text())
    assert complete['summary_sha256']==sha(run/'summary.json')
    assert complete['verification_sha256']==sha(run/'verification.json')
    verified=json.loads((run/'verification.json').read_text())
    assert verified['status']=='passed' and verified['units']==180
    summary=json.loads((run/'summary.json').read_text())
    lock=json.loads((run/'lock.json').read_text())
    assert len(lock['people'])==30
    for name in ['summary.json','verification.json','complete.json','lock.json','training_complete.json','RESULTS.md']:
        copy(run/name,HERE/'metrics'/name)
    source_files=['shared/models.py','modality_se/model.py','paired_se/model.py']
    for name in source_files: copy(V2/name,HERE/'implementation'/name)
    copy(V2/'rolling_personal_20261015/core.py',HERE/'implementation/rolling_core.py')
    copy(OLD/'implementation/requirements.txt',HERE/'implementation/requirements.txt')
    # All source checkpoints and all selected endpoint weights; never retune here.
    records=[]; selection_counts={}; n_support=n_test=0
    for mode in ['ppg','dual']:
        counts=collections.Counter()
        for seed in [42,43,44]:
            source=V2/'modality_se/models'/f'{mode}_pair_population_seed{seed}.pt'
            assert sha(source)==lock['spec']['source_checkpoints'][source.name]
            copy(source,HERE/'02_personalization/source_weights'/source.name)
            archive=HERE/'02_personalization/personal_weights'/f'{mode}_seed{seed}.tar.gz'
            archive.parent.mkdir(parents=True,exist_ok=True)
            with tarfile.open(archive,'w:gz') as tar:
                for entry in lock['people']:
                    pid=entry['patient'].replace(':','_')
                    unit=run/'units'/mode/f'seed{seed}'/pid
                    trained=json.loads((unit/'trained.json').read_text())
                    sel=json.loads((unit/'selection.json').read_text())
                    assert sel['final_test_used'] is False
                    if mode=='ppg' and seed==42:
                        n_support+=trained['n_support'];n_test+=trained['n_test']
                    for endpoint in [0,1]:
                        name=f'tail_endpoint{endpoint}.pt'
                        assert sha(unit/name)==trained['hashes'][name]
                        tar.add(unit/name,arcname=f'{pid}/{name}')
                        chosen=sel['endpoint_selections'][endpoint]['selected']['tail']
                        cfg=chosen['config'];counts[(endpoint, 'source' if cfg is None else str(cfg))]+=1
                        records.append(dict(modality=mode,seed=seed,patient=entry['patient'],endpoint=endpoint,
                            selection=chosen,n_support=trained['n_support'],n_test=trained['n_test'],
                            file=f'{archive.relative_to(HERE)}::{pid}/{name}',sha256=sha(unit/name)))
                    tar.add(unit/'selection.json',arcname=f'{pid}/selection.json')
        selection_counts[mode]=[dict(endpoint=d,configuration=c,count=n) for (d,c),n in sorted(counts.items())]
    write(HERE/'metrics/tail_selections.json',json.dumps(records,indent=2)+'\n')
    write(HERE/'metrics/tail_selection_counts.json',json.dumps(selection_counts,indent=2)+'\n')
    # Compact audit archive: predictions/labels, folds, selections, no raw signals.
    with tarfile.open(HERE/'metrics/unit_evidence.tar.gz','w:gz') as tar:
        for unit in sorted((run/'units').glob('*/*/*')):
            for p in sorted(unit.iterdir()):
                if p.suffix in ['.json','.npz']:
                    tar.add(p,arcname=str(p.relative_to(run/'units')))
    baseurl='https://github.com/place-bot/Bad-Blood/tree/research/v2-vitaldb-attention-personalization/model%20training/v2/final_results_20261015'
    intro='''# Bad Blood V2 - adopted personalization, 20261015

**The adopted personal method is PairSE tail adaptation with per-person,
per-endpoint hyperparameter selection.** Both PPG-only and ECG+PPG are retained.
The architecture is shared CNN + Squeeze-and-Excitation (SE) channel attention,
plus the reference-conditioned pressure-change head. Personal adaptation updates
the change head, final residual CNN block and its SE gate. Earlier blocks stay
frozen. SBP and DBP each use their own selected endpoint checkpoint.

Head-only adaptation and automatic head/tail scope selection were tested as
comparators. They are **not additional adopted models**. The experiment includes
a validation-selected zero-update source candidate; we preserve it as evaluated.

## Results on each person's later 20%

Thirty people; 8,882 personal development windows and 2,211 test windows.
Participant-macro MAE (mmHg), averaged across seeds 42, 43, 44, not an ensemble.
All methods use identical calibration and test records within each person.

| Input | Personal mean | Bias correction | Adopted tail adaptation | Reduction vs bias (SBP / DBP) |
|---|---:|---:|---:|---:|
'''
    for mode,label in [('ppg','PPG only'),('dual','ECG + PPG')]:
        r=summary['modality_results'][mode]['methods'];a=np.array(r['rolling_tail']['macro']['mae']);b=np.array(r['bias']['macro']['mae'])
        intro+=f'| {label} | {fmt(r["personal_mean"]["macro"]["mae"])} | {fmt(b)} | **{fmt(a)}** | {fmt((1-a/b)*100)}% |\n'
    intro+='''
Both endpoints improve over both controls in the **overall 30-person mean in
every seed**. Tail adaptation also has lower mean MAE than head-only adaptation.
This does not mean every individual improves: seed-averaged simultaneous wins
on both pressures against both controls are 10/30 (PPG) and 13/30 (dual).
This cohort was evaluated previously; this is an exploratory comparison, not
new independent external validation. The 20261013 results used different cohorts
and a smaller support budget; do not interpret their score difference as a
matched-method improvement. The current comparison re-evaluates the old fixed
method on the same 80/20 data as the new methods.

## Read and run

- [Current IEEE report](report/Bad_Blood_V2_Final_Report_20261015.pdf) and [LaTeX](report/main.tex).
- [Updated teaching Beamer](beamer/Bad_Blood_V2_Models_Explained.pdf) and [source](beamer/Bad_Blood_V2_Models_Explained.tex).
- [Population models](01_pretrained/README.md): unchanged source CNN+SE models and training records.
- [Adopted personal method](02_personalization/README.md): six source initializers,
  360 endpoint-specific personal checkpoints, and usage.
- [All methods, metrics, seeds and people](metrics/RESULTS.md), [machine-readable results](metrics/summary.json).
- [Portable fitting and inference](implementation/personalize.py).
- [Frozen full experiment code](../rolling_personal_20261015/), including head and scope-selection controls.
- [Independent experiment verification](metrics/verification.json) and [release checks](release_verification.json).

Only publicly sourced participant identifiers and derived experiment artifacts
are included. Raw PulseDB signals must be obtained from the official source;
there are no Notre Dame student records in this release. Saved personal weights
belong to the named participant; fit a fresh source copy for a new person.
The [20261013 snapshot](../final_results_20261013/) and V1 remain preserved.
'''
    write(HERE/'README.md',intro)
    write(HERE/'01_pretrained/README.md','''# Population models retained unchanged

Use the [CNN+SE architecture, six population checkpoints and training results](../../final_results_20261013/01_pretrained/README.md).
Source training: 1,085 VitalDB-derived PulseDB participants / 390,600 ten-second
windows; validation: 192 / 69,120. Main test: 141 / 56,400.
The 20261015 change is to **personal adaptation**, not new population training.
Both PPG-only and ECG+PPG remain available. The paired source initializers for
personalization are also copied into this release's `02_personalization/source_weights/`.
''')
    write(HERE/'02_personalization/README.md','''# Adopted V2 personal method: tail adaptation (20261015)

One policy, two inputs: PPG-only or ECG+PPG. The pair model predicts current
SBP/DBP from current signals plus this person's earlier reference signals and
known scalar BP. It does not need current true BP or an ABP waveform.

## Exactly what is fitted

- Trainable: `change.*`, `encoder.body.6.*`, `encoder.gates.6.*` (33,336 parameters).
  These are the pressure-change head, final residual CNN block, and last SE gate.
- Frozen: earlier encoder blocks, population readout weights, source label/feature
  normalization and BatchNorm running statistics. Dropout stays disabled.
  Trainable BatchNorm affine parameters inside the last block are retained.
- Adam; no weight decay; gradient norm capped at 5. Each update uses **all valid
  within-person chronological pairs**, with reference end + 10 seconds <= query start.
- Each person's first 80% of ordered eligible segments is development data.
  Last 20% is test, after a 10-second guard. Do not randomly shuffle this split.
- Within development, three expanding training/validation periods use
  0-50% -> 50-65%, 0-65% -> 65-80%, 0-80% -> 80-100%, with guards.
  These percentages refer to the development bank, not the full recording.
- Independently for SBP and DBP, choose learning rate {0.0001, 0.0003} and
  optimizer updates {10,25,50,100} by average validation MAE across the three periods.
  A zero-extra-update source checkpoint is also evaluated. Ties prefer fewer
  updates, then smaller learning rate. No other person's labels select these settings.
- Restart from the source checkpoint and refit on the entire personal development
  bank. Save **two endpoint models** per person/source seed. At inference, take SBP
  from endpoint 0 and DBP from endpoint 1, each averaged over all known references.

The `rolling_selected` experiment additionally allowed head-only versus tail
selection; that is a comparator, **not the adopted deployment policy**.
Population seed variation changes pretrained initialization/order/pair sampling,
not the 30 people, personal split or calibration budget. Results average three
separate runs; no ensemble is deployed by this package.

## Portable use

Run from `final_results_20261015/` (Python 3 + NumPy + PyTorch):

```sh
python -m pip install -r implementation/requirements.txt
python implementation/personalize.py fit \\
  --checkpoint 02_personalization/source_weights/ppg_pair_population_seed42.pt \\
  --support person_development_only.npz --output my_personal_model
python implementation/personalize.py predict \\
  --models my_personal_model --input later_windows.npy --output predictions.npy
```

`person_development_only.npz` contains `x` (K x 2 x 1250, ECG then PPG), `y`
(K x 2, SBP then DBP in mmHg), `start` and `end` (seconds in the same recording
clock). PPG-only also accepts K x 1 x 1250. Supply filtered, synchronized 125-Hz
signals, min-max normalized separately per channel/window. Reserve and guard
the final test period **before** creating this file. `fit` accepts no test data.
Its `reference_bank.npz` is required for later predictions; store it with the
person's models in controlled storage. Source seed 42 remains the representative
source choice from population validation, not a personal-test winner.

Six archives in `personal_weights/` contain the 30 people's two selected tail
endpoint checkpoints for each modality/seed (360 checkpoints total). Extract
the desired archive into a separate directory. These archives do not contain
raw reference waveforms. To replay an archived person, reconstruct their exact
reference bank using `metrics/lock.json` and supply `--references bank.npz`.
Never use one participant's personal model for a different person.
''')
    # Full-method report tables, no selective metric removal.
    tables={}
    for mode in ['ppg','dual']:
        r=summary['modality_results'][mode]['methods']
        tables[mode]=metric_table(r,['personal_mean','bias','rolling_tail'],rowlabels=['Personal mean','Bias correction','Tail adaptation'])
    make_report(summary, tables, selection_counts)
    update_beamer(beamer, summary, tables, selection_counts)
    print(json.dumps(dict(release=str(HERE),development=n_support,test=n_test,weights=len(records))))


def make_report(s, tables, counts):
    old=(OLD/'report/main.tex').read_text()
    pre=old.split(r'\begin{document}')[0].replace('20261013','20261015').replace('October 13','October 15')
    population=old[old.index(r'\section{Population architecture and training}'):old.index(r'\section{Reference-conditioned personalization}')]
    for name in ['epochs','architecture']:
        population=population.replace('\\input{tables/'+name+'}',(OLD/f'report/tables/{name}.tex').read_text())
    content=pre+r'''
\begin{document}\maketitle\thispagestyle{plain}
\begin{abstract}
Bad Blood V2 combines a compact population convolutional neural network with
Squeeze-and-Excitation (CNN+SE) and reference-conditioned PairSE personalization.
This 20261015 release adopts personal tail adaptation: the change head, last
residual CNN block and its SE gate are fitted separately for each participant.
Both photoplethysmogram-only (PPG) and electrocardiogram-plus-PPG (ECG+PPG)
versions are retained. Each person's first 80\% of ordered segments supplies
training and three temporal validation periods; their last 20\% is test.
Learning rate and update count are selected independently for systolic and
diastolic blood pressure (SBP/DBP). Across 30 participants and three source seeds,
participant-macro mean absolute error (MAE) is 6.89/4.29 mmHg for PPG and
7.14/4.31 for dual input. Relative to matched additive bias correction, these
are reductions of 19.0/20.3\% and 21.5/24.4\%. Both endpoints improve in the
overall mean in every seed. We retain complete controls, individual outcomes,
checkpoints and runnable code; the previously examined cohort makes this an
exploratory comparison rather than independent external confirmation.
\end{abstract}
\begin{IEEEkeywords}blood pressure, personalization, temporal validation,
PPG, ECG, CNN, Squeeze-and-Excitation\end{IEEEkeywords}

\section{Adopted method and scope}
The adopted personal method is \textbf{tail adaptation with person-specific,
endpoint-specific hyperparameters}. Single and dual signals are two input
versions of this one method. Head-only fitting and automatic selection between
head and tail were evaluated as controls, not retained as alternative deployment
policies. The population model and its source training remain unchanged.
The reported tail policy includes a validation-selected source checkpoint with
zero extra personal updates. This candidate is retained exactly as evaluated;
the fitting rules and recorded test outcomes are unchanged in this release.

\section{Data and matching}
All results use the VitalDB-derived portion of PulseDB \cite{wang2023pulsedb},
not the entire MIMIC-III/VitalDB collection or Notre Dame student recordings.
Source CNN+SE training used 1,085 people / 390,600 ten-second windows, validation
192 / 69,120 and main testing 141 / 56,400. Population main-test MAE remains
12.39/8.00 mmHg for ECG+PPG and 12.32/7.85 for PPG-only. These uncalibrated
population results are a different task and cohort from personal testing.

Personal evaluation randomly selected 30 of 39 eligible local individual records
using seed 20261015, excluding population train/validation/test participants
and the four previous personal-development people. The 30 people and their
indices are identical to the preceding personal-selection experiment.
The first 80\% of chronologically ordered eligible segments provides 8,882
development windows. A ten-second temporal guard precedes the final test period,
which contains 2,211 windows in total. Splits are not random within a person.
Each participant supplies the same development and test data to every control.
These target records have been examined in previous experiments; no final-test
labels entered this run's fitting or selection, but reuse limits confirmatory
interpretation. Earlier six-/sixteen-person, 20\%-support results remain in
the dated 20261013 archive and are not pooled with this new protocol.

Each input is a filtered, synchronized ten-second waveform at 125 Hz (1,250
samples/channel), min-max scaled independently per channel and segment.
ECG then PPG occupy the two input slots; PPG-only zeros the ECG slot throughout
source training, reference encoding and inference. PulseDB's ABP-derived scalar
SBP/DBP labels provide the targets. No current ABP waveform or current true BP
is an inference input. An earlier known scalar BP plus earlier signal is needed
for calibration; intermittent cuff labels are format-compatible but constitute
a separate, much sparser evaluation setting.
'''+population+r'''
\section{Reference-conditioned architecture}
PairSE shares one CNN+SE encoder between an earlier reference waveform and a
currently observed waveform. Its 48-dimensional features and two standardized
population predictions are denoted $h_r,p_r$ and $h_c,p_c$. With known reference
BP $y_r$, it constructs
\begin{align}
z_{cr}&=[h_c-h_r,h_r,y_r,p_c-p_r],\\
z_{rr}&=[0,h_r,y_r,0],\\
\widehat y_{c\mid r}&=y_r+(p_c-p_r)+g_\phi(z_{cr})-g_\phi(z_{rr}).
\end{align}
The pressure-change multilayer perceptron (MLP) is 100--64--32--2 with Sigmoid
Linear Unit (SiLU) activations and 8,610 parameters. Full PairSE has 46,906
parameters. Identity subtraction recovers the known BP exactly when both inputs
are identical. Final prediction averages over every known reference in that
person's development bank. Features and BP use distinct, source-training-only
normalization statistics. This is current-pressure estimation using observed
current signals, not prediction of an unseen future waveform.

The design adapts paired-reference and transfer-learning ideas
\cite{mekonnen2024,schlesinger2020,leitner2022}, rather than reproducing their
published networks. We combine a compact 1D residual encoder, a final-stage
SE gate, explicit reference BP, identity anchoring and all-reference averaging.
Group pair-head training freezes the encoder and trains on the same 390,600
source windows with random non-self references from the same participant.
It uses Adam, learning rate 0.001, batch 2,048, standardized mean squared error
(MSE), gradient clipping at 5, at most 20 epochs and patience 5. Source validation
selects checkpoints; target-person data is absent from this group training.

\section{Personal tail adaptation: exact pipeline}
\subsection{Trainable and frozen parameters}
The trainable components are \texttt{change.*}, \texttt{encoder.body.6.*} and
\texttt{encoder.gates.6.*}: the change head, last residual block and final SE
gate, totaling \textbf{33,336 parameters}. Earlier encoder blocks, population
readout weights, feature/BP normalization and BatchNorm running statistics
remain fixed. BatchNorm affine scale/bias inside the last block can change.
Dropout is disabled during personal fitting. The frozen population readout
can still produce different values because its input features are adapted.

\begin{figure}[t]\centering
\begin{tikzpicture}[font=\footnotesize,node distance=3mm,
box/.style={draw,align=center,text width=6.9cm,inner sep=4pt},>=Stealth]
\node[box] (a) {Current + reference signals from the same person};
\node[box,below=of a] (b) {Frozen stem + residual blocks 1 and 2};
\node[box,below=of b,fill=black!6] (c) {Trainable residual block 3 + SE channel gate};
\node[box,below=of c,fill=black!6] (d) {Trainable reference-conditioned change MLP};
\node[box,below=of d] (e) {Known reference BP + predicted change; bank average};
\draw[->](a)--(b);\draw[->](b)--(c);\draw[->](c)--(d);\draw[->](d)--(e);
\end{tikzpicture}
\caption{Adopted personal adaptation. Gray components update on one person's
data; early reference and current branches share the same weights.}
\end{figure}

\subsection{Only the person's first 80\% selects their settings}
For each person and source seed, create three expanding temporal folds within
their development bank: train 0--50\%, validate 50--65\%; train 0--65\%, validate
65--80\%; train 0--80\%, validate 80--100\%. These percentages are of the
\emph{development bank}, not the complete recording. Validation starts at least
10 seconds after the latest training window ends. The last 20\% of the full
record is never used in these folds. Candidate fitting always restarts from
the same source PairSE checkpoint.

For each endpoint separately, evaluate learning rates $\{10^{-4},3\times10^{-4}\}$
and update counts $\{10,25,50,100\}$, plus the source model with zero personal
updates. Minimize the equally weighted average of the three validation MAEs.
Ties prefer fewer updates, then smaller learning rate. SBP and DBP may select
different settings; another target person's data has no role in that choice.
After selection, restart from source and refit on the entire development bank.
There are two saved endpoint copies per participant/source seed. Predictions
take SBP from its selected copy and DBP from its selected copy, not both outputs
from an arbitrarily chosen checkpoint.

Every optimizer step uses all valid chronological reference/current pairs in
that fold's training bank, requiring reference end + 10 seconds $\leq$ current
start. The endpoint-specific objective is standardized MSE; the optimizer is
Adam, with no weight decay and gradient norm capped at 5. Update counts are
optimizer steps, not epochs. Pairing reuses observed labels and creates no new
independent observations. Frozen early features are cached; exact accumulated
pair gradients propagate through the trainable tail. Final testing freezes all
weights and uses only known development references and later signal inputs.

\subsection{Controls and source seeds}
Personal mean predicts the mean development BP. Additive bias correction is
$p(X_c)+|S|^{-1}\sum_{r\in S}[y_r-p(X_r)]$, using the same modality's population
model. It already tracks waveform-dependent changes, unlike the constant mean.
Other matched controls are the old fixed head-only PairSE (learning rate
0.0003, 25 joint-endpoint updates), previous single-period personal selection,
three-fold endpoint-specific head-only fitting, and validation-selected head/tail
scope. Only the tail candidate group is the adopted method.

Seeds 42/43/44 change the population training initialization, shuffled training
order and stochastic source pair-head training, yielding different initializers.
They do not change participants, temporal splits or reference budgets.
Personal fitting uses all valid pairs, rather than a newly randomized subset.
Three-seed averages are averages of separate evaluated runs, not an ensemble.

\section{Metrics and overall results}
We compute MAE, root MSE (RMSE), signed Bias, sample error standard deviation
(SD, denominator $n-1$), and fractions within 5, 10 and 15 mmHg, separately for
SBP/DBP. Participant-macro metrics give each of the 30 people equal weight;
pooled metrics give each test segment equal weight. Macro SD averages each
person's error SD and differs from pooled SD. All tables average three seeds.
Best values are bold within each table and endpoint; Bias is ranked by distance
from zero. Full per-person and per-seed metrics are available in the release.
'''
    for mode,label in [('ppg','PPG only'),('dual','ECG+PPG')]:
        r=s['modality_results'][mode]['methods'];a=np.array(r['rolling_tail']['macro']['mae']);b=np.array(r['bias']['macro']['mae']);m=np.array(r['personal_mean']['macro']['mae'])
        content+='\n\\subsection{'+label+'}\n'
        content+=f'Tail adaptation reaches {fmt(a)} mmHg MAE. Relative to personal mean, this is a reduction of {fmt(m-a)} mmHg ({fmt((1-a/m)*100)}\\%). Relative to additive bias correction, the reduction is {fmt(b-a)} mmHg ({fmt((1-a/b)*100)}\\%). Both endpoints improve in the overall mean in each of the three seeds.\n'
        content+=r'\begin{table}[ht]\centering\caption{'+label+r': participant-macro results. SBP / DBP.}'+tables[mode].replace(r'\smalltable',r'\scriptsize\setlength{\tabcolsep}{3pt}')+r'\end{table}'+'\n'
    content+=r'''
\subsection{What the tail and scope ablations show}
Compared with head-only adaptation, tail adaptation reduces macro MAE by
0.70/0.25 mmHg for PPG and 1.25/0.58 for dual input. PPG tail MAE varies across
seeds with SD 0.36/0.13 mmHg; dual tail SD is 0.11/0.11. PPG has slightly lower
mean error; dual has more consistent source-seed results. A paired participant
bootstrap interval for the PPG-minus-dual MAE difference crosses zero for both
endpoints, so these results do not establish a universal modality winner.

Automatic scope selection gives 7.15/4.23 (PPG) and 7.23/4.52 (dual). Tail-only
selection gives 6.89/4.29 and 7.14/4.31 respectively. Scope selection therefore
slightly improves PPG DBP but worsens PPG SBP and both dual endpoints. The
adopted policy fixes the trainable scope to the tail while retaining personal
selection of learning rate and update count. This release decision follows
the exploratory comparison, not an untouched external-test selection claim.

\subsection{Individual outcomes and uncertainty}
After averaging each person's errors across seeds, 10/30 PPG participants and
13/30 dual participants beat both controls on both pressures simultaneously.
The strict count in each seed is 11/9/7 (PPG) and 12/12/15 (dual); 3 and 10
people, respectively, win in every seed. These are secondary individual counts,
distinct from the overall mean, which improves in all six modality/seed runs.
Some large gains influence the mean. Excluding the largest beneficiary as a
descriptive sensitivity check still leaves tail-minus-bias MAE differences of
-0.85/-0.55 mmHg (PPG) and -1.33/-1.00 (dual). Official results retain all people.
The full report tables and machine-readable results preserve unfavorable
individual outcomes rather than claiming a benefit for every person.

\section{Implementation, artifacts and verification}
The current folder is \texttt{final\_results\_20261015}. The adopted method is
\texttt{rolling\_tail}. Six source PairSE initializers and 360 personal endpoint
checkpoints ($30\times3\times2\times2$) are provided, with all selections and
checksums. A portable fitting command accepts only one person's development
signals, known BP and timestamps. The inference command accepts current signals,
earlier references and the two endpoint checkpoints, never current query BP.
Raw reference signals are not redistributed; follow the frozen manifest to
recover the public source files. A personal checkpoint is not transferable to
another person without a fresh personal fit.

All 180 person/modality/seed units and 1,080 internal validation folds completed.
Independent verification replayed 720 endpoint models with maximum prediction
difference $1.53\times10^{-5}$ mmHg and confirmed matched budgets. The completed
record hashes match the archived summary and verification. This establishes
software/result consistency, not clinical certification. The earlier dated V2
results and V1 are preserved. The current portable release is additionally
checked against archived endpoint predictions and published with its own manifest.

\section{Conclusion}
V2 retains both population CNN+SE input modes and adopts reference-conditioned
tail personalization with independent within-person selection for each pressure.
The controlled 30-person comparison shows consistent overall MAE improvement
over personal mean and additive bias correction across all three seeds for both
modalities. Tail adaptation improves the overall result beyond head-only fitting.
The next independent evaluation should preserve this policy and assess new people,
including the distinct low-cost-sensor and cuff-reference setting at Notre Dame.
'''
    content+=r'''
\begin{thebibliography}{6}
\bibitem{wang2023pulsedb} W. Wang et al., ``PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation methods,'' Frontiers in Digital Health, vol. 4, 1090854, 2023. doi:10.3389/fdgth.2022.1090854.
\bibitem{hu2018se} J. Hu, L. Shen, and G. Sun, ``Squeeze-and-Excitation Networks,'' CVPR, pp. 7132--7141, 2018.
\bibitem{mekonnen2024} B. K. Mekonnen et al., ``Accurate and 30-plus days reliable cuffless blood pressure measurements with 9-minutes personal photoplethysmograph data and mixed deduction learning,'' Scientific Reports, vol. 14, 23722, 2024. doi:10.1038/s41598-024-75583-y; author correction doi:10.1038/s41598-024-79326-x.
\bibitem{schlesinger2020} O. Schlesinger et al., ``Estimation and Tracking of Blood Pressure Using Routinely Acquired Photoplethysmographic Signals and Deep Neural Networks,'' Critical Care Explorations, vol. 2, e0095, 2020.
\bibitem{leitner2022} J. Leitner, P.-H. Chiang, and S. Dey, ``Personalized Blood Pressure Estimation Using Photoplethysmography: A Transfer Learning Approach,'' IEEE JBHI, vol. 26, pp. 218--228, 2022.
\end{thebibliography}
\end{document}
'''
    content=content.replace(r'\end{document}', all_method_tables(s)+r'\end{document}')
    write(HERE/'report/main.tex',content)


def all_method_tables(s):
    out='\n\\clearpage\n\\onecolumn\n'
    for mode,label in [('ppg','PPG only'),('dual','ECG+PPG')]:
        if mode=='dual':out+='\n\\clearpage\n'
        out+=r'\section*{Complete matched comparisons: '+label+'}\n'
        r=s['modality_results'][mode]['methods']
        out+=r'\begin{table*}[ht]\centering\scriptsize\caption{'+label+r': all predeclared methods, participant-macro metrics. SBP / DBP.}'+ '\n'
        out+=r'\begin{tabular}{lrrrrrrr}\toprule Method & MAE & RMSE & Bias & Error SD & Within 5 (\%) & Within 10 (\%) & Within 15 (\%)\\\midrule'+'\n'
        for name,labelm in zip(METHODS,NAMES):
            cells=[]
            for m in METRICS:
                rank=lambda x: abs(x) if m=='bias' else -x if m.startswith('within') else x
                best=[min(rank(r[n]['macro'][m][d]) for n in METHODS) for d in [0,1]]
                vals=r[name]['macro'][m]
                cells.append(' / '.join((r'\textbf{'+f'{v*(100 if m.startswith("within") else 1):.2f}'+'}') if abs(rank(v)-best[d])<1e-10 else f'{v*(100 if m.startswith("within") else 1):.2f}' for d,v in enumerate(vals)))
            out+=labelm+' & '+' & '.join(cells)+r'\\'+'\n'
        out+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
        out+=r'\begin{table*}[ht]\centering\scriptsize\caption{'+label+r': seed-level participant-macro MAE; each column uses the same 30 people.}'+ '\n'+r'\begin{tabular}{lrrr}\toprule Method & Seed 42 & Seed 43 & Seed 44\\\midrule'+'\n'
        for n,l in zip(METHODS,NAMES):out+=l+' & '+' & '.join(fmt(r[n]['per_seed'][str(seed)]['macro']['mae']) for seed in [42,43,44])+r'\\'+'\n'
        out+=r'\bottomrule\end{tabular}\end{table*}'+'\n'
        out+=r'\begin{table*}[ht]\centering\scriptsize\caption{'+label+r': pooled segment-weighted controls and adopted model, SBP / DBP.}'+metric_table(r,['personal_mean','bias','rolling_tail'],mode='pooled',size=r'\scriptsize',rowlabels=['Personal mean','Bias correction','Tail adaptation'])+r'\end{table*}'+'\n'
    return out.replace(r'\begin{table*}[ht]',r'\begin{table}[!ht]').replace(r'\end{table*}',r'\end{table}')


def update_beamer(path, s, tables, counts):
    text=path.read_text()
    if 'Current personal results: 30 people' in text:
        copy(path,HERE/'beamer'/path.name)
        return
    # Preserve the existing open document, its teaching flow, figures and template.
    def replace(title, body):
        nonlocal text
        pat=r'\\begin\{frame\}\{'+re.escape(title)+r'\}.*?\\end\{frame\}'
        text,n=re.subn(pat,lambda _:frame(title,body).rstrip(),text,flags=re.S)
        assert n==1,(title,n)
    text=text.replace('October 13, 2026','October 15, 2026').replace('release 20261013','release 20261015').replace('Release snapshot 20261013','Release snapshot 20261015')
    text=text.replace('final_results_20261013','final_results_20261015')
    text=text.replace("Copy the shared model, then fit its change predictor to P","Copy the shared model, then fit its last CNN block, SE and change head to P")
    replace('The people and the jobs of their data',r'''
\whose{All groups come from the VitalDB-derived portion of PulseDB, not ND students.}
\smalltable\begin{tabular}{P{3.3cm}rrP{5.1cm}}\toprule
Group & People & Windows & Job\\\midrule
Group training & 1,085 & 390,600 & Learn shared weights\\
Group validation & 192 & 69,120 & Choose population checkpoints\\
Group main test & 141 & 56,400 & Score uncalibrated predictions\\
BP-range stress test & 116 & 666 & Different pressure range\\\midrule
Personal development & 30 & 8,882 & Each person's own early 80\%: train and select settings\\
Personal test & Same 30 & 2,211 & Each person's later 20\%: score only\\\bottomrule
\end{tabular}
\takeaway{Every target person selects settings using their own development data.}
\source{[7] A 10-second temporal guard removes boundary windows. Same 30 people across modes and seeds. These target records were previously examined; the current comparison is exploratory.}
''')
    replace("One person's timeline",r'''
\whose{Only person P. All 30 personal pipelines repeat this procedure independently.}
\begin{center}\begin{tikzpicture}[x=1cm,y=1cm]
\fill[Pale](0,0)rectangle(9.3,1.4);\fill[Navy!8](9.7,0)rectangle(12.4,1.4);
\node[align=center,text width=8.7cm]at(4.65,.7){P's first 80\% of ordered segments\\signals + known BP: training and internal validation};
\node[align=center,text width=2.4cm,font=\small]at(11.05,.7){Later 20\%\\test only};
\draw[Amber,thick](9.3,0)--(9.3,1.7);\draw[Amber,thick](9.7,0)--(9.7,1.7);
\node[above,font=\scriptsize]at(9.5,1.7){10-second guard};
\draw[conn](0,-.15)--(12.5,-.15);
\end{tikzpicture}\end{center}
\begin{itemize}\setlength\itemsep{3mm}
\item The \textbf{reference bank} stores P's development waveforms and known SBP/DBP.
\item Three time-ordered validation periods inside the 80\% choose P's settings.
\item After selection, refit on all of P's development bank, then freeze the model.
\end{itemize}
\source{[7] 20261015 protocol: chronological segment-count split, not a random split or the previous 20\%-of-elapsed-time support rule. Later true BP is available only to the scoring code.}
''')
    text=text.replace('then adapt the personal head','then adapt the personal tail and head')
    text=text.replace("The next stage instead makes 25 weight updates using \\textbf{one target person's early pairs}.","The next stage chooses the update count separately for each target person and pressure, using only their early data.")
    replace('Exactly what changes during personal fitting',r'''
\whose{Only this target person's development pairs; the starting weights come from the group.}
\smalltable\begin{tabular}{P{4.7cm}P{8cm}}\toprule
Component & During this person's fitting\\\midrule
Stem + residual blocks 1 and 2 & Frozen weights\\
Last residual CNN block & \textbf{Trainable} signal features\\
Last SE channel-attention gate & \textbf{Trainable} feature-channel weighting\\
Reference-conditioned change head & \textbf{Trainable} 100--64--32--2 network\\
Population readout weights & Frozen, but its input features can change\\
Label/feature scaling, BatchNorm statistics & Frozen; dropout disabled\\\bottomrule
\end{tabular}
\takeaway{33,336 trainable parameters per endpoint copy; 46,906 in full PairSE.}
\source{[7] Trainable names: change.*, encoder.body.6.*, encoder.gates.6.*. BatchNorm affine scale/bias in the last block can train. SE = Squeeze-and-Excitation; CNN = convolutional neural network.}
''')
    text=text.replace(r'\stage{Personal settings: chosen on 4 development people}',r'\stage{Personal settings: chosen separately within each person}')
    replace('Four development people choose the fitting settings',r'''
\whose{Only P's first 80\%. No other target person chooses P's hyperparameters.}
\smalltable\begin{tabular}{lrr}\toprule
Internal period & Train on development bank & Validate on development bank\\\midrule
1 & First 50\% & Next 15\% (50--65\%)\\
2 & First 65\% & Next 15\% (65--80\%)\\
3 & First 80\% & Last 20\% (80--100\%)\\\bottomrule
\end{tabular}
\par\medskip\textbf{These percentages are inside P's development bank.} P's final test 20\% is outside this entire procedure.
\begin{itemize}\setlength\itemsep{2mm}
\item Tail learning rate: \textbf{0.0001 or 0.0003}. Updates: \textbf{10,25,50,100}.
\item Also compare zero extra updates; every candidate starts from the source model.
\item Select SBP and DBP settings separately by mean MAE over the three periods.
\end{itemize}
\source{[7] Ten-second guards at validation boundaries. Ties favor fewer updates, then lower learning rate. Hyperparameters are settings, not fitted weights. MAE = mean absolute error.}
''')
    text=text.replace('Four development people choose the fitting settings','Each person chooses their own fitting settings')
    anchor=r'\stage{Stage 4: frozen personal model predicts P\'s later pressure}'
    extra=frame('From selected settings to a fitted personal model',r'''
\begin{enumerate}\setlength\itemsep{3mm}
\item Save P's selected learning rate and number of updates for \textbf{SBP}.
\item Save P's separately selected settings for \textbf{DBP}.
\item Restart two copies of the same source model. Fit each endpoint on \textbf{all of P's development bank} using its chosen settings.
\item Each Adam step uses \textbf{every valid chronological reference/current pair}; reference end + 10 seconds must precede current start.
\item Optimize that endpoint's standardized MSE; clip gradient norm at 5; no weight decay.
\end{enumerate}
\takeaway{Two endpoint checkpoints per person and source seed, using one tail-adaptation policy.}
\source{MSE = mean squared error. Adam = Adaptive Moment Estimation. The candidate with zero updates remains possible. The reference bank stays separate from the weights.}
''')
    # Insert before Stage 4 by its exact existing marker.
    marker="\\stage{Stage 4: frozen personal model predicts P's later pressure}"
    assert marker in text
    if 'From selected settings to a fitted personal model' not in text:text=text.replace(marker,extra+'\n'+marker)
    text=text.replace('We are not forecasting it from the first 20\\%.','The earlier 80\\% calibrates the model; the later signals are observed inputs.')
    # Historical worked arithmetic remains useful but is clearly separated from the new benchmark.
    text=text.replace('A real personal prediction, calculated step by step','A historical worked prediction: same PairSE equation')
    text=text.replace('Person \\texttt{p005879}, dual-input seed 42.', 'Historical 20261013 head-only example, person \\texttt{p005879}, dual-input seed 42.')
    text=text.replace('Replayed released personal checkpoint.', 'Replayed historical checkpoint, not a 20261015 tail-result example.')
    # Replace historical personal-result block with current evidence; preserve population slides.
    start=text.index(r'\stage{Personal results: fit each person early, score them later}')
    end=text.index(r'\stage{Transfer to Notre Dame: proposed next evaluation}')
    new=r'\stage{Current personal results: 30 people, later 20\% test}'+'\n'
    body=r'\small Equal-person MAE, then average of three seeds. Values are SBP / DBP, mmHg.\par\medskip\begin{tabular}{lrr}\toprule Method & PPG only & ECG+PPG\\\midrule'+'\n'
    for name,label in zip(METHODS,NAMES):
        vals=[fmt(s['modality_results'][m]['methods'][name]['macro']['mae']) for m in ['ppg','dual']]
        if name=='rolling_tail':vals=[r'\textbf{'+v+'}' for v in vals]
        body+=label+' & '+' & '.join(vals)+r'\\'+'\n'
    body+=r'\bottomrule\end{tabular}\takeaway{Adopted: tail adaptation, with each person selecting their own settings.}\source{Bold identifies the adopted method here; this is not a claim that it wins every endpoint. Scope selection has slightly lower PPG DBP MAE. All controls use matched calibration/test records.}'
    new+=frame('Matched comparison: all 30 people',body)
    rows=[]
    for mode,label in [('ppg','PPG only'),('dual','ECG+PPG')]:
        r=s['modality_results'][mode]['methods'];a=np.array(r['rolling_tail']['macro']['mae'])
        for base,bl in [('personal_mean','Personal mean'),('bias','Bias correction')]:
            b=np.array(r[base]['macro']['mae']);rows.append(label+' vs '+bl+' & '+fmt(b-a)+' & '+fmt((1-a/b)*100)+r'\%\\')
    new+=frame('How much does the adopted method improve?',r'\begin{center}\begin{tabular}{lrr}\toprule Comparison & MAE reduction (mmHg) & Relative reduction\\\midrule'+'\n'+'\n'.join(rows)+r'\bottomrule\end{tabular}\end{center}\takeaway{Both pressures improve in the overall 30-person mean in every source seed.}\small This is an overall result, not a guarantee for every participant.\source{Same participant weighting and same development/test records for each comparison; SBP / DBP.}')
    for mode,label in [('ppg','PPG only'),('dual','ECG+PPG')]:
        r=s['modality_results'][mode]['methods']
        new+=frame(label+': full metric panel',tables[mode]+r'\metricnote\takeaway{Tail adaptation improves both errors and coverage relative to matched controls.}\source{Thirty participants, three source seeds. Error SD is averaged within person; pooled results are reported separately in the IEEE report.}')
        body=r'\begin{center}\smalltable\begin{tabular}{lrrr}\toprule Method & Seed 42 & Seed 43 & Seed 44\\\midrule'+'\n'
        for m,l in [('personal_mean','Personal mean'),('bias','Bias correction'),('rolling_head','Head only'),('rolling_tail','Tail adaptation'),('rolling_selected','Scope selection')]:body+=l+' & '+' & '.join(fmt(r[m]['per_seed'][str(seed)]['macro']['mae']) for seed in [42,43,44])+r'\\'+'\n'
        body+=r'\bottomrule\end{tabular}\end{center}\takeaway{Tail adaptation beats both baselines on both pressures in every seed.}'
        body+=r'\small Tail MAE standard deviation across seeds: '+fmt(r['rolling_tail']['seed_sd']['mae'])+r' mmHg.\source{Seeds change source training randomness, not participants, splits or budgets. Results are separate runs, not an ensemble.}'
        new+=frame(label+': does the result repeat across seeds?',body)
    new+=frame('Tail adaptation versus choosing the scope',r'''
\textbf{Adopted method:} fix the trainable components to the change head + last CNN block + SE. Each person still selects learning rate and update count.
\par\medskip
\textbf{Scope-selection comparator:} additionally let internal validation choose head-only or tail fitting. This was tested, but is not our retained policy.
\par\medskip
\begin{tabular}{lrr}\toprule Policy & PPG MAE & Dual MAE\\\midrule
Head-only selection & 7.59 / 4.54 & 8.39 / 4.89\\
Tail selection (adopted) & 6.89 / 4.29 & 7.14 / 4.31\\
Head/tail scope selection & 7.15 / 4.23 & 7.23 / 4.52\\\bottomrule\end{tabular}
\takeaway{Personal hyperparameter selection does not require changing the trainable scope.}
\source{SBP / DBP, mmHg. Both strategies allow a validation-selected zero-update source checkpoint. The latter is a fallback, not a third learned architecture.}
''')
    new+=frame('Overall gains and individual outcomes answer different questions',r'''
\smalltable\begin{tabular}{lrr}\toprule Strict wins over both controls, both pressures & PPG & ECG+PPG\\\midrule
Seed 42 / 43 / 44 & 11 / 9 / 7 & 12 / 12 / 15\\
After averaging each person's MAE across seeds & 10 / 30 & 13 / 30\\
The same person wins in all three seeds & 3 / 30 & 10 / 30\\\bottomrule\end{tabular}
\par\medskip
\begin{itemize}\setlength\itemsep{3mm}
\item The overall mean improves in all seeds even though some individuals worsen.
\item Removing the largest beneficiary as a sensitivity check still gives tail-minus-bias MAE of -0.85/-0.55 (PPG), -1.33/-1.00 (dual).
\item The official results retain all 30 people; no test-based exclusion or retuning.
\end{itemize}
\source{Previously examined public-data cohort: exploratory evidence, not new external validation. Each person's full results remain in the release.}
''')
    text=text[:start]+new+'\n'+text[end:]
    text=text.replace('The public-data example used \\textbf{151 early labeled windows}.','Our adopted public-data experiment uses \\textbf{all eligible early-80\\% reference windows}.')
    replace('What we have, and what the results support',r'''
\begin{itemize}\setlength\itemsep{3mm}
\item \textbf{Population}: compact CNN+SE in PPG-only and ECG+PPG versions. Source training is unchanged.
\item \textbf{One personal policy}: update the change head, last CNN block and SE. Each person selects settings separately for SBP and DBP using three internal temporal periods.
\item \textbf{Overall personal MAE}: PPG 6.89/4.29; dual 7.14/4.31 mmHg. Both beat personal mean and bias correction in every seed.
\item \textbf{Retain both inputs}: PPG has slightly lower mean MAE; dual is more stable across source seeds. No universal input winner is established.
\end{itemize}
\par\medskip
\href{https://github.com/place-bot/Bad-Blood/tree/research/v2-vitaldb-attention-personalization/model\%20training/v2/final_results_20261015}{\textbf{20261015 release: code, endpoint weights, complete metrics and IEEE report}}
\source{[7] The source model, each person's two endpoint checkpoints, and their own reference bank define the personal predictor. Earlier releases remain archived.}
''')
    path.write_text(text)
    copy(path,HERE/'beamer'/path.name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--beamer',type=Path,required=True)
    a=p.parse_args();publish(a.run,a.beamer)
