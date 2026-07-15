# Latent-Variable Sampling for a Simplex-Valued Posterior

## 1. Problem setting

Let

$$
\boldsymbol{\alpha}
=
\bigl(\alpha(1),\ldots,\alpha(Y)\bigr)
$$

be a probability vector satisfying

$$
\alpha(y)\ge 0,
\qquad
\sum_{y=1}^Y \alpha(y)=1.
$$

Assume a Dirichlet prior

$$
p(\boldsymbol{\alpha})
\propto
\prod_{y=1}^Y \alpha(y)^{\beta_y-1},
\qquad
\beta_y>0,
$$

and the posterior log-density

$$
\log p(\boldsymbol{\alpha}\mid D)
=
\sum_{y=1}^Y(\beta_y-1)\log \alpha(y)
+
\sum_{i=1}^m
\log
\left[
\sum_{y=1}^Y
\frac{p_{\mathrm{tr}}(y\mid x_i)}
     {p_{\mathrm{tr}}(y)}
\alpha(y)
\right]
+
\text{const}.
$$

Define

$$
c_{iy}
=
\frac{p_{\mathrm{tr}}(y\mid x_i)}
     {p_{\mathrm{tr}}(y)}.
$$

The posterior density can then be written as

$$
p(\boldsymbol{\alpha}\mid D)
\propto
\left[
\prod_{y=1}^Y \alpha(y)^{\beta_y-1}
\right]
\left[
\prod_{i=1}^m
\sum_{y=1}^Y c_{iy}\alpha(y)
\right].
$$

This posterior is generally **not Dirichlet**, because the likelihood contains products of sums rather than only powers of the components $\alpha(y)$.

---

## 2. Introducing latent class variables

For every observation $x_i$, introduce a latent variable

$$
z_i\in\{1,\ldots,Y\}.
$$

The variable $z_i$ indicates which class contributes to the likelihood term associated with $x_i$.

Define the augmented joint density

$$
p(\boldsymbol{\alpha},\boldsymbol{z}\mid D)
\propto
\left[
\prod_{y=1}^Y \alpha(y)^{\beta_y-1}
\right]
\left[
\prod_{i=1}^m c_{i,z_i}\alpha(z_i)
\right],
$$

where

$$
\boldsymbol{z}=(z_1,\ldots,z_m).
$$

This construction is useful because summing over all possible values of the latent variables recovers the original posterior:

$$
\begin{aligned}
\sum_{\boldsymbol{z}}
p(\boldsymbol{\alpha},\boldsymbol{z}\mid D)
&\propto
\left[
\prod_{y=1}^Y \alpha(y)^{\beta_y-1}
\right]
\prod_{i=1}^m
\sum_{z_i=1}^Y c_{i,z_i}\alpha(z_i)
\\
&=
\left[
\prod_{y=1}^Y \alpha(y)^{\beta_y-1}
\right]
\prod_{i=1}^m
\sum_{y=1}^Y c_{iy}\alpha(y).
\end{aligned}
$$

Thus, the latent-variable model is exactly equivalent to the original posterior after marginalization over $\boldsymbol{z}$.

---

## 3. Conditional distribution of the latent variables

Conditioned on $\boldsymbol{\alpha}$, the latent variables are independent:

$$
p(\boldsymbol{z}\mid\boldsymbol{\alpha},D)
=
\prod_{i=1}^m p(z_i\mid\boldsymbol{\alpha},x_i).
$$

For a particular observation $i$,

$$
p(z_i=y\mid\boldsymbol{\alpha},x_i)
\propto
c_{iy}\alpha(y).
$$

After normalization,

$$
p(z_i=y\mid\boldsymbol{\alpha},x_i)
=
\frac{c_{iy}\alpha(y)}
     {\sum_{k=1}^Y c_{ik}\alpha(k)}.
$$

In the original notation,

$$
p(z_i=y\mid\boldsymbol{\alpha},x_i)
=
\frac{
\alpha(y)
\frac{p_{\mathrm{tr}}(y\mid x_i)}
     {p_{\mathrm{tr}}(y)}
}{
\sum_{k=1}^Y
\alpha(k)
\frac{p_{\mathrm{tr}}(k\mid x_i)}
     {p_{\mathrm{tr}}(k)}
}.
$$

Therefore, every $z_i$ can be sampled from a categorical distribution with probabilities

$$
r_{iy}
=
\frac{c_{iy}\alpha(y)}
     {\sum_{k=1}^Y c_{ik}\alpha(k)}.
$$

---

## 4. Conditional distribution of the probability vector

For fixed latent assignments $\boldsymbol{z}$, define the class counts

$$
n_y
=
\sum_{i=1}^m \mathbf{1}[z_i=y].
$$

The terms involving $\boldsymbol{\alpha}$ in the augmented posterior are

$$
\begin{aligned}
p(\boldsymbol{\alpha}\mid\boldsymbol{z},D)
&\propto
\prod_{y=1}^Y \alpha(y)^{\beta_y-1}
\prod_{i=1}^m \alpha(z_i)
\\
&=
\prod_{y=1}^Y
\alpha(y)^{\beta_y+n_y-1}.
\end{aligned}
$$

This is a Dirichlet density:

$$
\boxed{
\boldsymbol{\alpha}\mid\boldsymbol{z},D
\sim
\operatorname{Dirichlet}
\bigl(
\beta_1+n_1,\ldots,\beta_Y+n_Y
\bigr).
}
$$

The latent-variable augmentation therefore restores conditional conjugacy.

---

## 5. Gibbs sampling algorithm

The two conditional distributions lead directly to a Gibbs sampler.

### Initialization

Choose an initial probability vector

$$
\boldsymbol{\alpha}^{(0)}
\in\Delta^{Y-1},
$$

for example

$$
\alpha^{(0)}(y)=\frac{1}{Y}.
$$

### Iteration $t=1,\ldots,T$

#### Step 1: Sample latent assignments

For each observation $i=1,\ldots,m$, compute

$$
r_{iy}^{(t)}
=
\frac{
c_{iy}\alpha^{(t-1)}(y)
}{
\sum_{k=1}^Y
c_{ik}\alpha^{(t-1)}(k)
}.
$$

Then sample

$$
z_i^{(t)}
\sim
\operatorname{Categorical}
\left(
r_{i1}^{(t)},\ldots,r_{iY}^{(t)}
\right).
$$

#### Step 2: Compute class counts

For every class $y$, calculate

$$
n_y^{(t)}
=
\sum_{i=1}^m
\mathbf{1}[z_i^{(t)}=y].
$$

#### Step 3: Sample the probability vector

Sample

$$
\boldsymbol{\alpha}^{(t)}
\sim
\operatorname{Dirichlet}
\left(
\beta_1+n_1^{(t)},\ldots,
\beta_Y+n_Y^{(t)}
\right).
$$

After discarding an initial burn-in period, the samples

$$
\boldsymbol{\alpha}^{(B+1)},\ldots,
\boldsymbol{\alpha}^{(T)}
$$

approximate samples from the posterior $p(\boldsymbol{\alpha}\mid D)$.

---

## 6. Pseudocode

```text
Input:
    c[i,y] = p_tr(y | x_i) / p_tr(y)
    prior parameters beta[1:Y]
    number of iterations T

Initialize:
    alpha[y] = 1 / Y

For t = 1,...,T:

    For i = 1,...,m:
        For y = 1,...,Y:
            weight[y] = c[i,y] * alpha[y]

        probabilities = weight / sum(weight)
        z[i] ~ Categorical(probabilities)

    For y = 1,...,Y:
        n[y] = number of indices i such that z[i] = y

    alpha ~ Dirichlet(beta + n)

Return the stored alpha samples after burn-in.
```

---

## 7. Why the augmentation works

The original likelihood contribution of observation $x_i$ is

$$
\sum_{y=1}^Y c_{iy}\alpha(y).
$$

This can be interpreted as marginalizing over a hidden categorical choice $z_i$:

$$
\sum_{y=1}^Y
p(z_i=y\mid\boldsymbol{\alpha})\,
p(x_i\mid z_i=y),
$$

up to factors absorbed into $c_{iy}$.

Conditioned on $z_i$, the likelihood contribution becomes proportional to a single component $\alpha(z_i)$. The complete-data likelihood is consequently multinomial in $\boldsymbol{\alpha}$, and the Dirichlet prior is conjugate to it.

The sampler alternates between:

1. assigning each observation probabilistically to a latent class;
2. sampling the class-probability vector from the resulting Dirichlet posterior.

---

## 8. Mixture-of-Dirichlet interpretation

The posterior can also be understood as a finite mixture of Dirichlet distributions.

Expanding the product gives

$$
\prod_{i=1}^m
\left(
\sum_{y=1}^Y c_{iy}\alpha(y)
\right)
=
\sum_{\boldsymbol{z}\in\{1,\ldots,Y\}^m}
\left[
\prod_{i=1}^m c_{i,z_i}
\right]
\prod_{y=1}^Y\alpha(y)^{n_y(\boldsymbol{z})}.
$$

Therefore,

$$
p(\boldsymbol{\alpha}\mid D)
=
\sum_{\boldsymbol{z}}
w_{\boldsymbol{z}}\,
\operatorname{Dirichlet}
\left(
\boldsymbol{\alpha};
\boldsymbol{\beta}
+
\boldsymbol{n}(\boldsymbol{z})
\right),
$$

for suitable normalized mixture weights $w_{\boldsymbol{z}}$.

There are potentially $Y^m$ latent assignments, so explicitly constructing this mixture is usually infeasible. Gibbs sampling avoids enumerating all mixture components and instead explores them stochastically.

---

## 9. Computational complexity

Computing all assignment probabilities requires approximately

$$
O(mY)
$$

operations per Gibbs iteration.

Sampling the Dirichlet vector costs only $O(Y)$. Therefore, the dominant cost is the latent-label update.

For $Y\le 10$, this method is computationally attractive even for a reasonably large number of observations.

---

## 10. Numerical implementation notes

### Work with logarithms

If some $c_{iy}$ or $\alpha(y)$ are very small, compute

$$
\log w_{iy}
=
\log c_{iy}+\log\alpha(y)
$$

and normalize using the log-sum-exp operation.

### Zero probabilities

If $c_{iy}=0$, then class $y$ has zero conditional probability for observation $i$. This is valid, provided that at least one $c_{iy}$ is positive for every $i$.

### Positive prior parameters

The Dirichlet parameters must satisfy

$$
\beta_y>0.
$$

Using $\beta_y<1$ favors sparse probability vectors near the simplex boundary, while $\beta_y>1$ favors vectors away from the boundary.

### Convergence diagnostics

Useful diagnostics include:

- trace plots of each $\alpha(y)$;
- autocorrelation plots;
- effective sample size;
- multiple-chain comparison using $\widehat{R}$;
- posterior summaries from several dispersed initializations.

---

## 11. Possible collapsed sampler

It is also possible to integrate out $\boldsymbol{\alpha}$ analytically and sample only the latent variables. Under the Dirichlet prior,

$$
p(z_i=y\mid\boldsymbol{z}_{-i},D)
\propto
c_{iy}
\left(
\beta_y+n_{-i,y}
\right),
$$

where $n_{-i,y}$ is the number of observations assigned to class $y$, excluding observation $i$.

After sampling the latent assignments, one obtains a posterior sample of the probability vector from

$$
\boldsymbol{\alpha}
\sim
\operatorname{Dirichlet}
\left(
\boldsymbol{\beta}+\boldsymbol{n}
\right).
$$

The collapsed sampler can sometimes mix faster because it removes the dependence between consecutive samples of $\boldsymbol{\alpha}$, although each latent-variable update depends on the current assignment counts.

---

## 12. Summary

The posterior

$$
p(\boldsymbol{\alpha}\mid D)
\propto
\prod_y\alpha(y)^{\beta_y-1}
\prod_i\sum_y c_{iy}\alpha(y)
$$

is generally not a single Dirichlet distribution. However, introducing latent categorical variables $z_i$ makes the conditional posterior of $\boldsymbol{\alpha}$ Dirichlet:

$$
\boldsymbol{\alpha}\mid\boldsymbol{z},D
\sim
\operatorname{Dirichlet}
\left(
\boldsymbol{\beta}+\boldsymbol{n}
\right).
$$

This yields a simple Gibbs sampler that alternates between:

$$
z_i\mid\boldsymbol{\alpha},D
\sim
\operatorname{Categorical}
\left(
\frac{c_{iy}\alpha(y)}
     {\sum_k c_{ik}\alpha(k)}
\right)
$$

and

$$
\boldsymbol{\alpha}\mid\boldsymbol{z},D
\sim
\operatorname{Dirichlet}
\left(
\boldsymbol{\beta}+\boldsymbol{n}
\right).
$$

For a small number of classes, such as $Y\le 10$, this is a natural and efficient exact MCMC method for sampling from the posterior.
