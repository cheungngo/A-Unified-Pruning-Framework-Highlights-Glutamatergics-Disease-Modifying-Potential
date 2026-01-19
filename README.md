# Modeling Antidepressant-Induced Manic Switch and Longitudinal Relapse: A Unified Pruning Framework Highlights Glutamatergics' Disease-Modifying Potential


Authors:

Ngo Cheung, FHKAM(Psychiatry)

Affiliations:

¹ Independent Researcher

Corresponding Author:

Ngo Cheung, MBBS, FHKAM(Psychiatry)

Hong Kong SAR, China

Tel: 98768323

Email: info@cheungngomedical.com

**Conflict of Interest**: None declared.

**Funding Declaration**: This research received no specific grant from
any funding agency in the public, commercial, or not-for-profit sectors.

**Ethics Declaration**: Not applicable.

Citation:
Cheung, N. (2026). Modeling Antidepressant-Induced Manic Switch and Longitudinal Relapse: A Unified Pruning Framework Highlights Glutamatergics' Disease-Modifying Potential. Zenodo. https://doi.org/10.5281/zenodo.18298989


## Abstract

Background: Major depressive disorder involves impaired neural
plasticity, yet antidepressants targeting glutamatergic (ketamine),
monoaminergic (SSRIs), and GABAergic (neurosteroids) pathways differ
markedly in onset speed, durability, and risk of treatment-emergent
mania---particularly in bipolar contexts. Clinical comparisons are
confounded by heterogeneity; computational models enable controlled
mechanistic dissection, but few integrate manic liability and
post-discontinuation stability across classes.

Methods: We extended a magnitude-based pruning model (95% sparsity) of
depression in feed-forward networks classifying Gaussian blobs. From
identical pruned baselines, three interventions were simulated:
ketamine-like gradient-guided synaptic regrowth (50% reinstatement) with
consolidation; SSRI-like prolonged low-rate refinement with tapering
noise and escalating excitability gain; neurosteroid-like global tonic
inhibition (0.7× damping, tanh activations, reduced gain). Efficacy
assessed classification accuracy under clean, noisy, and combined
stress; resilience via graded noise tolerance; acute relapse after
further pruning; manic risk through biased positive perturbation and
activation magnitude. Longitudinal relapse modeled chronic maintenance
(with mood stabilizer protection) followed by discontinuation, using
treatment-specific lingering decay rates. Metrics averaged across 10
seeds.

Results: All treatments restored near-ceiling performance acutely, but
ketamine-like regrowth yielded superior extreme-stress resilience
(76.8%) and zero post-discontinuation manic relapse, reducing sparsity
to 47.5%. Neurosteroid-like modulation matched rapid recovery (97.6%)
but showed state-dependence and 88.3% relapse probability off-drug.
SSRI-like refinement lagged in resilience (49.9% extreme) with highest
manic proxies (biased accuracy 47.2%, gain 1.60) and 95.0% relapse
post-cessation. Longer maintenance conferred negligible added protection
for reversible mechanisms.

Conclusions: Antidepressants operate via divergent plasticity
routes---durable structural rebuilding (ketamine-like, low long-term
risk), rapid reversible stabilization (neurosteroid-like), and
vulnerable gradual optimization (SSRI-like)---reproducing clinical
trade-offs in speed, persistence, and bipolar safety. These findings
support mechanism-guided selection, positioning synaptogenic agents for
recurrent or high-risk cases pursuing remission beyond treatment.

## Introduction

Major depressive disorder (MDD) is a leading contributor to disability
worldwide and imposes a substantial burden on individuals, families, and
health-care systems \[1\]. Contemporary pharmacotherapy has helped many
patients, yet full remission after an initial antidepressant trial is
achieved in only about one-third of cases, and many people remain
symptomatic despite several treatment attempts \[2\]. Selective
serotonin re-uptake inhibitors (SSRIs) typically require several weeks
before benefits become obvious, leaving patients exposed to prolonged
distress and only partial relief \[3\].

The delayed and incomplete response seen with SSRIs has fuelled interest
in compounds that act on other signalling systems. Low-dose ketamine, an
NMDA-receptor antagonist, can lift mood within hours and appears to do
so by stimulating brain-derived neurotrophic factor (BDNF) and
mTOR-dependent synaptogenesis \[4,5\]. Neuroactive steroids such as
zuranolone also show rapid antidepressant effects especially for
postpartum depression \[6\]. These findings have shifted attention away
from a strictly monoaminergic model toward the view that MDD involves
impaired neural plasticity, in which chronic stress erodes dendritic
spines and synaptic density in cortical and hippocampal regions \[7\].

Another clinical complication is treatment-emergent mania, particularly
in bipolar disorder, where conventional antidepressants provoke mood
switches in roughly 20--40 % of patients \[8\]. Initial findings
indicate that ketamine presents a reduced acute switch risk in
controlled settings \[9\], while preliminary studies suggest a
negligible risk associated with neurosteroids \[6\]. It is crucial to
comprehend how these mechanistically distinct treatments affect
depressive symptoms and the excitatory--inhibitory balance; however,
direct clinical comparisons are impeded by variability in patient
populations, dosing regimens, and concurrent therapies.

Computational modelling offers a controlled way to disentangle these
factors. Prior pruning-based simulations have cast depression as a state
of excessive synaptic loss, with ketamine-like regrowth restoring
network resilience. Few studies, however, have placed glutamatergic,
monoaminergic, and GABAergic strategies side by side or explored how
they affect risks such as manic switching or post-treatment stability.

The present work addresses these gaps through an extended
magnitude-pruning model applied to feed-forward neural networks.
Beginning with identical over-pruned networks, we simulated three
treatment motifs: ketamine-like synaptogenesis, SSRI-like gradual
refinement accompanied by rising excitability, and neurosteroid-like
tonic inhibition. End-points included acute antidepressant efficacy,
resilience to stress, proxies for manic conversion (biased excitatory
challenge and activation amplitude), immediate relapse risk, and---for
the first time in this framework---long-term vulnerability after chronic
maintenance and full discontinuation, incorporating treatment-specific
wash-out profiles. By embedding these elements in one plasticity-centred
model we aim to clarify the trade-offs in speed, durability, and bipolar
safety across drug classes, thereby informing mechanism-based treatment
selection.

## Methods

### Network architecture and classification task

Figure 1 shows the experimental workflow for multi-Mechanism
antidepressant comparison. We used a small feed-forward network to stand
in for key cortico-limbic circuits. The model had two input units, three
hidden layers of 512, 512 and 256 units, and a four-unit soft-max output
layer. Hidden layers normally used ReLU activation; during neurosteroid
simulations tanh was substituted to mimic the ceiling effect of tonic
GABAergic currents. Altogether the network held about 3.9 × 10\^5
trainable weights.

The learning task was simple four-way pattern recognition.
Two-dimensional points were drawn from four Gaussian clouds centred at
(−3, −3), (−3, 3), (3, −3) and (3, 3) with a standard deviation of 0.8.
Each run used 12 000 labelled training points. Three test sets were
prepared:

- a 4 000-item set with the same noise as the training data (standard
  > condition);

- a 2 000-item noise-free set (clean condition);

- perturbed versions for stress tests, created by adding extra Gaussian
  > noise after each hidden layer or by multiplying all activations with
  > a global gain factor.

A \"mood-stabiliser\" guard--rail was implemented as three scalar
parameters: a protection level (0--1) that capped gain, a small
inhibitory bias (−0.15 × protection) and a factor (0.3) that reduced the
upward shift of internal noise. All code ran in PyTorch on a single CPU.
Ten different random seeds (affecting data order and weight
initialisation) produced ten independent \"subjects.\"

![](media/image2.png){width="3.6806999125109363in"
height="7.931899606299212in"}

***Figure 1.** Experimental Workflow for Multi-Mechanism Antidepressant
Comparison. The pipeline consists of three distinct phases: (1) Baseline
Pathology Generation, where a neural network is trained and pruned to
simulate a depressed, sparse state; (2) Acute Treatment, where the model
branches into three mechanistic arms---Ketamine (structural regrowth),
SSRI (functional gain increase), and Neurosteroids (inhibition
modulation)---to reverse pathology; and (3) Longitudinal Manic Relapse
Simulation, where treated models undergo a maintenance phase followed by
medication discontinuation. In this final phase, residual mood
stabilizer protection decays at treatment-specific rates, and the model
is subjected to a biased noise trigger test to assess the probability of
manic switching.*

### Simulation of the depressive state

The network was first trained for 20 epochs with Adam (learning rate
0.001) on noise-free data until it reached ceiling accuracy (Figure 1).
Depression was then modelled by iterative magnitude pruning: across the
three hidden layers the 95 % smallest-magnitude weights were zeroed,
leaving a sparse and fragile network. Clean-input accuracy stayed high,
but performance collapsed when noise or further pruning was applied,
mirroring the stress sensitivity of a depressed brain \[7\].

### Antidepressant treatment protocols

From the same pruned starting point three treatment routines were run on
separate copies.

Ketamine-like: A modest global gain (1.25) was fixed. Gradients were
collected over 30 mini-batches to locate strong silent synapses; half of
these pruned weights were re-instated with small random values drawn
from N(0, 0.03). Fifteen fine-tuning epochs followed (Adam, 0.0005)
while the mask was locked.

SSRI-like: Sparsity (95 %) was left unchanged. Over 100 epochs the
internal noise level fell linearly from 0.5 to 0, while the global gain
rose from 1.0 to 1.6, simulating slow monoaminergic adaptation. Learning
used Adam with a rate of 1 × 10\^-5.

Neurosteroid-like: We kept the prune mask but multiplied post-activation
values by 0.7, switched ReLU to tanh, and set the gain at 0.85
(effective ≈0.59). Ten consolidation epochs (Adam, 0.0005) followed.

### Mood-stabiliser extension and longitudinal relapse test

After acute treatment, a chronic phase was added. A full protection
level (1.0) was turned on and held during maintenance, then allowed to
decay after drug discontinuation. Decay rates were treatment-specific:
0.002 per step for the ketamine model (long-lasting structural change),
0.015 for the SSRI model (rapid reversal) and 0.008 for the neurosteroid
model (intermediate). Maintenance lasted 25, 50, 100, 150, 200 or 300
low-rate epochs (Adam, 1 × 10\^-6). Drugs were then removed in one step,
and 50 decay steps were run to wash out residual protection. Manic risk
was probed by injecting strongly positive internal noise (σ = 1.0, shift
= +1.0); relapse was logged when accuracy fell below 60 %.

### Outcome measures

Primary efficacy was the percentage of correct classifications on clean,
standard-noise and combined-stress test sets. Resilience curves were
built by repeating tests with internal noise ranging from 0 to 2.5.
Acute relapse was tested by pruning a further 40 % of the remaining
weights and retesting under combined stress.

Manic conversion risk was indexed two ways: accuracy under highly biased
noise (lower accuracy = higher risk) and the mean absolute activation in
hidden layers (higher activation = greater latent excitability).
Neurosteroid state-dependence was recorded as the drop in accuracy when
the damping module was turned off.

## Results

Simulations were repeated with ten different random seeds, and every
outcome followed the same rank order across seeds, indicating that the
findings are robust to stochastic variation in data shuffling and weight
initialisation.

### Acute antidepressant efficacy

Before treatment, the heavily pruned network managed only 29.7 ± 2.7 %
accuracy when clean inputs were combined with internal and external
noise. Introducing any of the three treatment routines produced a
dramatic rebound (Table 1). Neurosteroid-like damping lifted
combined-stress accuracy to 97.6 ± 0.3 %, ketamine-like synaptogenesis
to 97.2 ± 0.2 %, and SSRI-like refinement to 90.5 ± 3.0 %. On both the
noise-free and the standard-noise test sets all treated models reached
or approached ceiling performance, whereas the untreated model stayed
near one-third correct. The ketamine condition achieved its improvement
with an effective sparsity of 47.5 %, reflecting reinstated connections;
the other two conditions retained the original 95 % sparsity.

***Table 1.** Antidepressant Efficacy (Mean ± SD Across 10 Seeds)*

| **Treatment**      | **Sparsity (%)** | **Clean (%)** | **Standard (%)** | **Combined (%)** |
|--------------------|------------------|---------------|------------------|------------------|
| Untreated (pruned) | 95.0 ± 0.0       | 34.7 ± 11.9   | 36.8 ± 11.9      | 29.7 ± 2.7       |
| Ketamine-like      | 47.5 ± 0.0       | 100.0 ± 0.0   | 100.0 ± 0.0      | 97.2 ± 0.2       |
| SSRI-like          | 95.0 ± 0.0       | 100.0 ± 0.0   | 99.9 ± 0.1       | 90.5 ± 3.0       |
| Neurosteroid-like  | 95.0 ± 0.0       | 100.0 ± 0.0   | 100.0 ± 0.0      | 97.6 ± 0.3       |

### 

### Stress resilience

Performance was next examined while internal Gaussian noise was
increased stepwise from none to a standard deviation of 2.5 (Table 2).
Ketamine-treated networks tolerated the severest disturbance best,
holding 76.8 ± 3.6 % accuracy at the highest noise level. SSRI-treated
networks fell to 49.9 ± 2.8 %, and neurosteroid-treated networks to 43.0
± 1.0 %. At moderate and high noise (σ = 1.0--1.5) the ketamine and
neurosteroid models performed similarly (93.0--98.2 %) and both
outperformed the SSRI model (64.9--78.4 %). The untreated network
hovered around 30 % regardless of noise intensity.

***Table 2**. Stress Resilience Profile (Mean ± SD Across 10 Seeds)*

| **Treatment**      | **None (%)** | **Moderate (σ=0.5) (%)** | **High (σ=1.0) (%)** | **Severe (σ=1.5) (%)** | **Extreme (σ=2.5) (%)** |
|--------------------|--------------|--------------------------|----------------------|------------------------|-------------------------|
| Untreated (pruned) | 36.8 ± 11.9  | 29.9 ± 2.5               | 29.6 ± 1.8           | 29.8 ± 1.4             | 29.6 ± 1.5              |
| Ketamine-like      | 100.0 ± 0.0  | 99.9 ± 0.1               | 98.2 ± 1.1           | 92.9 ± 2.4             | 76.8 ± 3.6              |
| SSRI-like          | 99.9 ± 0.1   | 95.1 ± 3.2               | 78.4 ± 5.4           | 64.9 ± 5.3             | 49.9 ± 2.8              |
| Neurosteroid-like  | 100.0 ± 0.0  | 99.9 ± 0.1               | 93.0 ± 2.6           | 70.6 ± 2.9             | 43.0 ± 1.0              |

### Manic conversion risk

Potential switch liability was probed with strongly positive, biased
internal noise (Table 3). The SSRI routine, which had wound excitability
up to a gain of 1.60, proved most vulnerable: biased-noise accuracy
averaged 47.2 ± 12.7 %, and the mean absolute hidden-layer activation
was 0.390 ± 0.079. Neurosteroid modulation, despite lowering gain to
0.85, achieved only slightly better biased-noise accuracy (50.6 ± 7.9 %)
and showed the lowest activation magnitude (0.196 ± 0.008). Ketamine
treatment combined a moderate gain of 1.25 with markedly safer
behaviour, sustaining 84.2 ± 8.5 % accuracy under the same biased
challenge and exhibiting the highest activation magnitude (0.649 ±
0.079) without instability. The pruned, untreated model remained both
hypo-active (0.100 ± 0.013) and inaccurate (25.0 ± 0.8 %).

***Table 3**. Manic Conversion Risk Metrics (Mean ± SD Across 10 Seeds)*

| **Treatment**      | **Gain Multiplier** | **Biased Stress Accuracy (%)** | **Activation Magnitude** |
|--------------------|---------------------|--------------------------------|--------------------------|
| Untreated (pruned) | 1.00 ± 0.00         | 25.0 ± 0.8                     | 0.100 ± 0.013            |
| Ketamine-like      | 1.25 ± 0.00         | 84.2 ± 8.5                     | 0.649 ± 0.079            |
| SSRI-like          | 1.60 ± 0.00         | 47.2 ± 12.7                    | 0.390 ± 0.079            |
| Neurosteroid-like  | 0.85 ± 0.00         | 50.6 ± 7.9                     | 0.196 ± 0.008            |

*Note. Lower biased stress accuracy indicates higher manic conversion
vulnerability; higher activation magnitude reflects greater latent
hyperexcitability.*

### Acute relapse vulnerability

Durability was tested by excising a further 40 % of the remaining
weights after treatment. Ketamine-treated networks were essentially
unaffected, their combined-stress accuracy changing by −0.1 ± 0.3 %.
Neurosteroid-treated networks lost 5.1 ± 2.1 % and SSRI-treated networks
7.0 ± 2.4 %, confirming a clear advantage for the structural regrowth
produced by the ketamine routine.

### Neurosteroid medication dependence

To gauge state-dependence, the neurosteroid damping module was switched
off after the acute phase. Combined-stress accuracy fell from 97.6 ± 0.3
% to 78.5 ± 4.9 %, and biased-noise accuracy dropped from 50.6 ± 7.9 %
to 36.9 ± 9.6 %. Interestingly, accuracy at the most extreme unbiased
noise level (σ = 2.5) rose from 43.0 ± 1.0 % to 58.3 ± 4.1 %, indicating
that tonic inhibition trades robustness to excitation-biased threats for
reduced tolerance of diffuse noise.

### Longitudinal manic relapse after discontinuation

A chronic maintenance phase was appended, followed by complete drug
withdrawal and gradual decay of the virtual mood-stabiliser. Decay rates
were set a priori to 0.002 per step for ketamine-treated networks, 0.015
for SSRI-treated networks, and 0.008 for neurosteroid-treated networks.
After all durations of maintenance (25--300 additional training epochs)
and the full wash-out period, ketamine-treated networks never relapsed:
biased-noise accuracy remained above 91 % in every seed. In contrast,
SSRI-treated networks relapsed in 95 % of all seed-by-duration
combinations, and neurosteroid-treated networks in 88.3 %.
Post-withdrawal biased-noise accuracy for the SSRI and neurosteroid
groups stabilised in the low-forties, irrespective of how long
maintenance had lasted, whereas the ketamine group stayed in the
low-nineties. These observations confirm that the protective changes
induced by the ketamine routine are both structurally persistent and
highly effective at preventing manic-like destabilisation, while the
functional adaptations driven by SSRIs and the partially state-dependent
modulation produced by neurosteroids leave the system vulnerable once
the drugs and the auxiliary stabiliser are withdrawn (Table 4).

***Table 4.** Summary Comparison Matrix*

| **Metric**              | **Ketamine-like** | **SSRI-like** | **Neurosteroid-like** | **Untreated** |
|-------------------------|-------------------|---------------|-----------------------|---------------|
| Combined Stress (%)     | 97.2              | 90.5          | 97.6                  | 29.7          |
| Biased Stress (%)       | 84.2              | 47.2          | 50.6                  | 25.0          |
| Gain Multiplier         | 1.25              | 1.60          | 0.85                  | 1.00          |
| Activation Magnitude    | 0.649             | 0.390         | 0.196                 | 0.100         |
| Acute Relapse Drop (%)  | −0.1              | 7.0           | 5.1                   | N/A           |
| Manic Relapse Prob. (%) | 0.0               | 95.0          | 88.3                  | N/A           |
| MS Decay Rate           | 0.0020            | 0.0150        | 0.0080                | N/A           |

## Discussion

### Interpretation of acute and resilience findings

The three simulated treatment paths behaved much as clinicians might
expect at the bedside. When the pruned network was exposed to
simultaneous external and internal noise---our analogue of depressive
pressure---both the ketamine-like and neurosteroid-like routines snapped
performance back to almost normal within a few training steps. The
slower, SSRI-like schedule helped, but never quite caught up. This
mirrors the clinic, where ketamine can lift mood in hours \[4\] and
zuranolone in a few days \[10\], whereas selective-serotonin reuptake
inhibitors usually need several weeks \[3\].

Differences emerged when we kept turning up the internal noise. Networks
that had undergone ketamine-style synaptogenesis kept working even at
the most extreme setting, a result that fits reports of durable stress
buffering after ketamine-induced structural change \[5\].
Neurosteroid-like damping steadied the system only as long as the
inhibitory module stayed in place; once removed, performance slid,
echoing the clinical observation that benefits from a short zuranolone
course can wane \[11\]. SSRI-like refinement offered the least cushion,
tracking the modest resilience frequently seen when conventional
antidepressants are the sole therapy in difficult cases \[2\].

### Manic conversion risk and excitability balance

Our proxies for switch risk told a familiar story. Raising network gain
in the SSRI-like condition produced the greatest drop in accuracy when a
positive noise bias---our stand-in for incipient mania---was introduced.
The result parallels the 20--40 % switch rate associated with
antidepressants in bipolar disorder \[8\]. The ketamine-like model
showed only a mild gain increase yet kept biased-noise accuracy high,
consistent with the low switch rates reported when ketamine is used
alongside mood stabilizers \[9\]. The neurosteroid-like routine lowered
both gain and hidden-unit activation and therefore looked safest,
matching early reports that zuranolone rarely provokes mania \[6,12\].

These patterns underline how the route to recovery shapes the
excitation--inhibition balance. Building new synapses tolerates some
extra excitatory drive; tonic inhibition suppresses it outright; simply
turning up global gain, as with the SSRI model, risks overshoot unless
other brakes are applied.

### Long-term stability after discontinuation

When we added a maintenance phase and then withdrew all drugs, the
contrasts sharpened. Circuits repaired in the ketamine-like way never
relapsed---even after the mood-stabiliser parameters had almost fully
decayed---suggesting that structural change can make the system
self-supporting. Clinical series describing months-long benefit after
limited ketamine infusions in bipolar depression point in the same
direction \[13\]. By comparison, nearly every SSRI-like or
neurosteroid-like network relapsed once protective settings were lifted,
regardless of how long maintenance had lasted. Naturalistic studies show
a similar pattern: recurrence remains common after antidepressant or
neurosteroid withdrawal, often exceeding 40 % a year \[14\]. Extending
maintenance did not help these two models much, mirroring data that slow
tapers reduce but do not remove relapse risk when monoaminergic drugs
are stopped \[15\]. Only the strategy that rebuilt connectivity
fundamentally altered vulnerability.

### Implications for clinical judgment and treatment selection

The simulation highlights how different drug mechanisms may guide
day-to-day prescribing (Table 5). Circuits rebuilt through a
ketamine-like process kept their stability long after the \"drug\" was
withdrawn, suggesting that glutamatergic agents could suit patients who
want lasting relief without continuous medication. That property is
already seen in clinical series where repeated ketamine infusions
provide benefit for months in otherwise resistant depression \[13\].

By contrast, the model that mimicked SSRI action showed the highest risk
of a manic switch and the greatest relapse once treatment stopped. These
results echo long-standing warnings about monoaminergic monotherapy in
people with bipolar features and about the sharp rise in recurrence
after antidepressant discontinuation \[14\]. When such agents are used,
combining them with lithium or valproate and planning for ongoing
maintenance remain prudent steps \[15\].

Neurosteroid-like modulation offered fast symptom control and little
acute excitability, consistent with early zuranolone studies in
postpartum and bipolar depression \[6\]. Yet the same model relapsed
quickly once the inhibitory drive was removed, implying that these drugs
may work best as short bridges---useful in urgent situations, but
followed by a hand-off to treatments that remodel the network more
permanently.

Taken together, the findings support a tiered strategy (Table 5).
Plasticity-inducing drugs may be chosen for chronic, relapsing, or
bipolar-spectrum illness where durable remission is the goal; GABAergic
neurosteroids can fill short-term needs for rapid relief; and
traditional antidepressants should be reserved for well-selected
unipolar cases or used with sturdy mood-stabilizing partners. Matching a
patient\'s history of switches, relapse density, and treatment goals to
these distinct profiles could improve outcomes as new rapid-acting
options become available.

***Table 5:** How Simulation Results May Inform Clinical Judgment in
Antidepressant Selection*

<table>
<colgroup>
<col style="width: 15%" />
<col style="width: 14%" />
<col style="width: 17%" />
<col style="width: 14%" />
<col style="width: 17%" />
<col style="width: 20%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Antidepressant Class (Model Analog)</strong></th>
<th><strong>Acute Efficacy &amp; Speed</strong></th>
<th><strong>Durability &amp; Post-Discontinuation
Stability</strong></th>
<th><strong>Manic Conversion Risk (Acute &amp;
Longitudinal)</strong></th>
<th><strong>Recommended Clinical Contexts</strong></th>
<th><strong>Key Considerations from Model</strong></th>
</tr>
<tr class="odd">
<th><p><strong>Ketamine-like</strong></p>
<p>(Glutamatergic synaptogenesis)</p></th>
<th>Rapid, near-complete recovery (97.2% combined stress)</th>
<th>Highest resilience to extreme stress (76.8%); zero manic relapse
post-cessation; structural changes persist</th>
<th>Moderate acute (biased accuracy 84.2%); lowest long-term
vulnerability</th>
<th><p>• Treatment-resistant unipolar depression</p>
<p>• Recurrent or bipolar-spectrum illness aiming for remission beyond
ongoing treatment</p>
<p>• Patients seeking stability without indefinite medication</p></th>
<th>Prioritize for cases needing disease-modifying potential; supports
earlier escalation in refractory depression</th>
</tr>
<tr class="header">
<th><p><strong>SSRI-like</strong></p>
<p>(Monoaminergic refinement + gain escalation)</p></th>
<th>Slower/incomplete (90.5% combined stress)</th>
<th>Lowest resilience (49.9% extreme stress); near-certain relapse
post-cessation (95%)</th>
<th>Highest acute (biased accuracy 47.2%, gain 1.60) and longitudinal
risk</th>
<th><p>• Low manic-switch-risk unipolar depression</p>
<p>• Only with indefinite mood stabilization in bipolar</p></th>
<th>Avoid monotherapy in bipolar vulnerability; requires concurrent mood
stabilizers and careful monitoring due to rapid recurrence upon
cessation</th>
</tr>
<tr class="odd">
<th><p><strong>Neurosteroid-like</strong></p>
<p>(GABAergic tonic inhibition)</p></th>
<th>Rapid, near-complete recovery (97.6% combined stress)</th>
<th>Moderate resilience (43.0% extreme stress); high relapse
post-cessation (88.3%); state-dependent</th>
<th>Lowest acute (biased accuracy 50.6%, damped excitability)</th>
<th><p>• Urgent scenarios needing rapid relief (e.g., postpartum
depression, acute bipolar depressive episodes)</p>
<p>• Short-term bridging</p></th>
<th>Excellent for acute safety and speed; use as bridge with planned
tapering or transition to more durable agents</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

*Note. This table distills the model\'s comparative profiles into
actionable guidance, emphasizing mechanism-based stratification.
Clinicians should integrate patient-specific factors (e.g., prior switch
history, episode density) alongside these insights when selecting or
sequencing treatments.*

### Novelty and potential impact

![](media/image1.png){width="6.268099300087489in"
height="5.1944991251093615in"}

***Figure 2.** Methodological novelty and clinical implications of the
unified computational framework. Unlike previous studies that modeled
antidepressant mechanisms in isolation, this framework initializes three
distinct pharmacological pathways---synaptogenic, monoaminergic, and
GABAergic---from a shared, severely pruned network baseline. By uniquely
tracking treatment-emergent mania and post-withdrawal relapse, the model
differentiates between agents that offer structural \"circuit reserve\"
(ketamine-like) versus those providing symptomatic relief with high
discontinuation fragility (SSRI-like and neurosteroid-like). These
divergent outcomes provide a mechanistic logic for future patient
stratification and clinical trial design.*

This study is one of the few attempts to place three very different
antidepressant mechanisms inside a single, carefully controlled
computational frame (Figure 2). Most earlier models concentrated on one
pathway at a time---for example, pruning models that mimic synaptic loss
in depression \[7\] or simulations of ketamine-driven regrowth alone
\[16\]. By contrast, the present work starts every network from the same
severely pruned state and then applies, side-by-side, a ketamine-like
synaptogenic routine, an SSRI-like slow gain adaptation, and a
neurosteroid-like tonic inhibition. In doing so it also tracks outcomes
that matter for bipolar illness---treatment-emergent mania and relapse
after drug withdrawal---areas that computational studies usually ignore.

The resulting picture is clinically recognizable. Ketamine-style
regrowth stands out for long-term stability; once new connections form,
the model keeps its resilience even when medication parameters decay.
This finding echoes emerging clinical views of ketamine as more than a
symptomatic drug, possibly a disease-modifying agent in hard-to-treat or
bipolar depression \[5,13\]. In contrast, both the SSRI-like and
neurosteroid-like routes restore function quickly but leave the system
fragile once treatment stops, mirroring high relapse rates seen after
discontinuing these medications \[14\] and the need for bridging
strategies after short neurosteroid courses \[11\]. By embedding all
three paths in the same architecture the model offers a clear,
mechanistic logic that could guide future guidelines, trial design, and
biomarker work---for example, identifying patients with low \"circuit
reserve\" who might benefit most from synaptogenic drugs.

### Limitations

Several simplifications temper direct clinical translation. The
feed-forward network omits the recurrent and oscillatory loops that
dominate cortico-limbic mood circuits, so real-world instabilities may
be underestimated. All interventions were applied globally, whereas in
vivo actions are cell-type and region specific---especially for
extrasynaptic GABA-A targets of neurosteroids \[12\]. Manic risk was
approximated by adding biased noise, not by modelling full affective
episodes, and subject variability was limited to random seeds rather
than patient-like heterogeneity in pruning depth or plasticity reserve.
Finally, mood-stabilizer co-therapy was represented only by simple decay
rates; explicit multi-drug interactions were not explored. These choices
kept the comparison tractable but mark priorities for future recurrent,
multi-compartment, or spiking models.

### Conclusion

Placing synaptogenesis, tonic inhibition, and gradual gain tuning on the
same depleted substrate clarifies how each path balances speed,
durability, and bipolar safety. Structural rebuilding---our ketamine
analogue---alone provides lasting resilience; reversible GABAergic
damping and slow monoaminergic tuning deliver rapid relief but require
continuing support in vulnerable patients. By moving beyond
transmitter-specific narratives toward concepts of circuit reserve and
excitability control, the model offers a practical framework for
personalised antidepressant selection as rapid-acting options continue
to grow.

## References

\[1\] World Health Organization. (2022). World mental health report:
Transforming mental health for all. World Health Organization.

\[2\] Rush, A. J., Trivedi, M. H., Wisniewski, S. R., et al. (2006).
Acute and longer-term outcomes in depressed outpatients requiring one or
several treatment steps: A STAR\*D report. American Journal of
Psychiatry, 163(11), 1905--1917.
https://doi.org/10.1176/ajp.2006.163.11.1905

\[3\] Trivedi, M. H., Rush, A. J., Wisniewski, S. R., et al. (2006).
Evaluation of outcomes with citalopram for depression using
measurement-based care in STAR\*D: Implications for clinical practice.
American Journal of Psychiatry, 163(1), 28--40.
https://doi.org/10.1176/appi.ajp.163.1.28

\[4\] Murrough, J. W., Iosifescu, D. V., Chang, L. C., et al. (2013).
Antidepressant efficacy of ketamine in treatment-resistant major
depression: A two-site randomized controlled trial. American Journal of
Psychiatry, 170(10), 1134--1142.
https://doi.org/10.1176/appi.ajp.2013.13030392

\[5\] Krystal, J. H., Abdallah, C. G., Sanacora, G., et al. (2019).
Ketamine: A paradigm shift for depression research and treatment.
Neuron, 101(5), 774--778. https://doi.org/10.1016/j.neuron.2019.02.005

\[6\] Gunduz-Bruce, H., Lasser, R., Nandy, I., et al. (2020, September).
Open-label, Phase 2 trial of the oral neuroactive steroid GABAA receptor
positive allosteric modulator zuranolone in bipolar disorder I and II.
In Poster presented at: psych Congress.

\[7\] Duman, R. S., & Aghajanian, G. K. (2012). Synaptic dysfunction in
depression: Potential therapeutic targets. Science, 338(6103), 68--72.
https://doi.org/10.1126/science.1222939

\[8\] Tondo, L., Vázquez, G., & Baldessarini, R. J. (2010). Mania
associated with antidepressant treatment: comprehensive meta‐analytic
review. Acta Psychiatrica Scandinavica, 121(6), 404-414.

\[9\] Jawad, M. Y., et al. (2021). Ketamine for bipolar depression: A
systematic review. International Journal of Neuropsychopharmacology, 24,
535--541. https://doi.org/10.1093/ijnp/pyab023

\[10\] Deligiannidis, K. M., Meltzer-Brody, S., Gunduz-Bruce, H., et al.
(2021). Effect of zuranolone vs placebo in postpartum depression: A
randomized clinical trial. JAMA Psychiatry, 78(9), 951--959.
https://doi.org/10.1001/jamapsychiatry.2021.1559

\[11\] Price, M. Z., & Price, R. L. (2025). Zuranolone for Postpartum
Depression in Real-World Clinical Practice. J Clin Psychiatry, 86(3),
25cr15876.

\[12\] Marecki, R., Kałuska, J., Kolanek, A., et al. (2023).
Zuranolone--synthetic neurosteroid in treatment of mental disorders:
narrative review. Frontiers in Psychiatry, 14, 1298359.

\[13\] Fancy, F., Rodrigues, N. B., Di Vincenzo, J. D., et al. (2023).
Real-world effectiveness of repeated ketamine infusions for
treatment-resistant bipolar depression. Bipolar disorders, 25(2),
99--109. https://doi.org/10.1111/bdi.13284

\[14\] Vázquez, G. H., Holtzman, J. N., Lolich, M., et al. (2015).
Recurrence rates in bipolar disorder: systematic comparison of long-term
prospective, naturalistic studies versus randomized controlled trials.
European Neuropsychopharmacology, 25(10), 1501-1512.

\[15\] Viktorin, A., Lichtenstein, P., Thase, M. E., et al. (2014). The
risk of switch to mania in patients with bipolar disorder during
treatment with an antidepressant alone and in combination with a mood
stabilizer. American Journal of Psychiatry, 171(10), 1067-1073.

\[16\] Cheung, N. (2026). Divergent mechanisms of antidepressant
efficacy: A unified computational comparison of synaptogenesis,
stabilization, and tonic inhibition in a model of depression
\[Preprint\]. Zenodo. https://doi.org/10.1281/zenodo.18290014
