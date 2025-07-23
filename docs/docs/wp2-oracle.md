# Discriminative risk (DR)


We propose a fairness quality measure named *discriminative risk (DR)* to reflect both individual and group fairness aspects. We also investigate its properties and establish the first and second-order oracle bounds concerning fairness to show that fairness can be boosted via ensemble combination with theoretical learning guarantees. The analysis is suitable for both binary and multi-class classification. Furthermore, an ensemble pruning method named *POAF (Pareto Optimal Ensemble Pruning via Improving
Accuracy and Fairness Concurrently)* is also proposed to utilise DR. Comprehensive experiments are conducted to evaluate the effectiveness of the proposed methods.

The full paper entitled *Increasing Fairness via Combination with Learning Guarantees* can be found on [arXiv](https://arxiv.org/pdf/2301.10813). There are also a couple of short versions for dissemination purposes only, see a non-archival [document](https://openreview.net/pdf?id=QHILhNkVUX) and its [poster](https://eustomadew.github.io/posters/2024_m3l_bounds.pdf), as well as [slides'23](https://eustomadew.github.io/slides/pre23_letall.pdf) and [slides'24](https://eustomadew.github.io/slides/pre24_melanie.pdf).


## Methodology

**Discriminative risk (DR)**

Following the principle of individual fairness, *the treatment/evaluation of one instance should not change solely due to minor changes in its sensitive attributes* (sen-att-s, aka. protected attributes). If it happens, this indicates the existence of underlying *discriminative risks*.

Naturally, the *fairness quality* of one hypothesis $f(\cdot)$ can be evaluated by

$$ \ell_\text{bias}(f,\mathbf{x})= \mathbb{I}(\overbrace{ f(\breve{\mathbf{x}}, \mathbf{a})\neq f(\breve{\mathbf{x}}, \underbrace{ \tilde{\mathbf{a}} }_{\hbox{ slightly perturbed version of sen-att-s }} ) }^{ \hbox{$f$ makes a discriminative decision} }) \,,$$

evaluating the risk from an individual aspect, and the empirical DR over one dataset describes this from a group aspect, as an unbiassed estimation of the true DR over one data distribution. There are no restrictions applying to the type of $f(\cdot)$.


**Oracle bounds and PAC bounds regarding fairness for the weighted voting**

If the weighted vote makes a discriminative decision, then *at least a $\rho$-weighted half* of the individual classifiers *have made a discriminative decision* and, therefore, the DR of an ensemble can be bounded by a constant times the DR of the individual classifiers. In other words, there exists a cancellation-of-biases effect in combination, similar to its well-known cancellation-of-error effect. We also provided two PAC bounds regarding fairness to bound the discrepancy between one hypothesis (either an individual classifier or an ensemble)'s empirical DR and its true DR.


**POAF (Pareto Optimal Ensemble Pruning via Improving
Accuracy and Fairness Concurrently)**



## Usage examples
<!-- Examples of how to use them -->


## Empirical result reproduction