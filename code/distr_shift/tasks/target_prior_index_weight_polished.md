# Target prior by class weights: define the test prior from a subset of classes

## Goal

Extend `run_real_reject_option_exp.py` with a fourth way to specify the
target label prior: the user names a **subset of classes and their relative
weights**, and a single **fallback weight** for every class not named. This
is convenient when only a few classes should be up- or down-weighted and the
rest are meant to stay uniform among themselves — expressing that today
requires writing out all `Y` entries of `--test-prior` by hand.

## Notation

Let `Y` be the number of classes and let the user give

- a sequence of `k` distinct class indices $y_1, \ldots, y_k$ with
  $1 \le k \le Y$ and $y_i \in \{0, \ldots, Y-1\}$,
- their weights $w_1, \ldots, w_k$ with $w_i \ge 0$,
- a fallback weight $R \ge 0$ applied to each class **not** in the sequence.

Note that $w_i$ is indexed by *position in the sequence*, not by class
index. Define the per-class weight $\tilde{w} : \{0,\ldots,Y-1\} \to
\mathbb{R}_{\ge 0}$ by

$$
\tilde{w}(c) =
\begin{cases}
w_i & \text{if } c = y_i \text{ for some } i \in \{1,\ldots,k\},\\
R   & \text{otherwise.}
\end{cases}
$$

The target prior is the normalized weight vector

$$
p(c) = \frac{\tilde{w}(c)}{Z},
\qquad
Z = \sum_{c=0}^{Y-1} \tilde{w}(c) = \sum_{i=1}^{k} w_i + (Y-k)\,R .
$$

`R` is a **per-class** weight, not a total mass to be shared: each of the
`Y - k` unnamed classes receives weight `R` individually, so unnamed classes
are always equiprobable among themselves. (This differs deliberately from
`--pair-rest-ratio`, whose second number is a *total* mass split
proportionally to the training prior.)

When `k = Y` every class is named, `R` contributes nothing to `Z`, and the
formula reduces to $p(c) = \tilde{w}(c) / \sum_i w_i$.

## Command-line interface

- `--prior-classes Y1 [Y2 ...]` (int, one or more) — the class indices
  $y_1,\ldots,y_k$.
- `--prior-weights W1 [W2 ...]` (float, one or more) — the weights
  $w_1,\ldots,w_k$, positionally aligned with `--prior-classes`.
- `--prior-rest-weight R` (float) — the per-class weight of the unnamed
  classes. Required when `k < Y`; **no default** (a silent default of 0
  would place zero mass on the unnamed classes, and a default of 1 would
  silently pick a scale for the user).

The three flags form one strategy: `--prior-classes` and `--prior-weights`
must be given together. Class indices are 0-based, matching
`--confusable-pair`.

Example — CIFAR-10, put four times the mass on classes 3 and 5 relative to
each of the other eight:

```
--prior-classes 3 5 --prior-weights 4 4 --prior-rest-weight 1
```

## Validation

Exit with a clear error message (`sys.exit("error: ...")`, as elsewhere in
the script) on any of:

1. `--prior-classes` given without `--prior-weights`, or vice versa.
2. `len(--prior-classes) != len(--prior-weights)`.
3. An index outside `[0, Y)`, or a repeated index.
4. Any $w_i < 0$, or $R < 0$.
5. `k < Y` and `--prior-rest-weight` not given.
6. `Z = 0` (all weights zero, so the prior is undefined).

Two further cases are **not** errors but must print a warning, mirroring how
the script already warns about degenerate `--pair-rest-ratio` settings:

- `k = Y` and `--prior-rest-weight` was passed — it is ignored.
- Some class receives weight 0 — it will be absent from the adaptation and
  evaluation sets.

Note that a zero-mass class is a hard error in *dirichlet mode*: the existing
check that every class has positive central mass (`Dir(s p)` is improper
otherwise) applies unchanged to a prior built this way, and its message
should mention the new flags alongside `--test-prior` / `--pair-rest-ratio`.

## Conflicting strategies

The script now offers four mutually exclusive ways to build the target
prior:

1. the registry/`--confusable-pair` default (with `--pair-ratio`,
   `--pair-rest-ratio`),
2. `--test-prior`,
3. the new `--prior-classes` / `--prior-weights` / `--prior-rest-weight`,
4. (nothing given) the training prior with the registry confusable pair
   re-weighted.

Passing flags from more than one of groups 2 and 3, or combining either with
an **explicitly passed** flag from group 1, exits with an error naming both
offending flags. This replaces the current behaviour, where `--test-prior`
colliding with the pair flags only prints a warning — make that case an
error too, for consistency.

"Explicitly passed" is essential here and must not be inferred from the
parsed value: `--confusable-pair` falls back to a registry default and
`--pair-ratio` has a non-`None` default of `(1.0, 7.0)`, so a value-based
check would report a conflict on every dataset that has a registry pair.
Detect it by giving the group-1 conflict-relevant flags a sentinel default of
`None`, or by inspecting `sys.argv` / an `argparse` sentinel.

## Interaction with dirichlet mode

Without `--dirichlet`, the resulting `p` *is* the target prior. With
`--dirichlet s`, it plays the same role as any other configured prior: it
becomes the **central** prior of `Dir(s * p)` from which each repetition's
target prior is drawn, and (unless `--beta` overrides it) the model prior
`s * p`. No special-casing is needed beyond the positive-mass check above.

## Reporting

The run report already prints a `confusable :` line describing how the
target prior was built. When the new strategy is used, print instead a line
naming the source and the resolved prior, e.g.

```
target prior : class weights  cat=4, dog=4, rest=1 (per class)
```

using `class_names` for readability, and keep the existing full target-prior
vector dump unchanged.
