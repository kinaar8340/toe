# \(S^2\) drift chord of `golden_rotate_cli`

```
MODEL, not theorem.
Geometry and arithmetic stay on QGA / flux_hopf_lib.
The alkane / CH2 / carbene language is an analogy for a discrete insertion step.
n=1 is one flywheel at quaternion identity (methane slot).
n=2 is one extra published step: one extra flywheel XOR one extra rotor insertion.
Do not emit “proves”, “element”, “periodic table identity”, or “carbene is a flywheel”.
```

One-step \(S^2\) geometry of the published axis rule. analog: not \(n\), not \(Z\), not an alkane identity.

Let bake-\(x\) be the unit axis \(e_x\) and let \(v\in S^2\) be the CLI axis. One published step acts on the axis by the rotation \(R_{e_x}(\theta)\). The \(S^2\) drift of that step is the geodesic length of the chord from \(v\) to \(R_{e_x}(\theta)v\):

\[
\delta(v,\theta)
=\arccos\bigl(v\cdot R_{e_x}(\theta)v\bigr).
\]

**Rodrigues about \(e_x\).**
Write \(v\) in components parallel and perpendicular to \(e_x\):

\[
v=\cos\varphi\,e_x+\sin\varphi\,u,
\qquad
u\perp e_x,\quad |u|=1,
\]

where \(\varphi=\arccos(v\cdot e_x)\) is the angle between the CLI axis and bake-\(x\). Then

\[
R_{e_x}(\theta)v
=\cos\varphi\,e_x
+\sin\varphi\bigl(\cos\theta\,u+\sin\theta\,(e_x\times u)\bigr).
\]

**Dot product.**
\(e_x\cdot u=0\) and \(e_x\cdot(e_x\times u)=0\), so

\begin{align*}
v\cdot R_{e_x}(\theta)v
&=
(\cos\varphi\,e_x+\sin\varphi\,u)
\cdot
\bigl(\cos\varphi\,e_x+\sin\varphi\cos\theta\,u+\sin\varphi\sin\theta\,(e_x\times u)\bigr)\\
&=
\cos^2\varphi
+\sin^2\varphi\cos\theta.
\end{align*}

The cross term with \(e_x\times u\) drops because \(u\perp(e_x\times u)\). Hence

\[
\delta(v,\theta)
=\arccos\bigl(\cos^2\varphi+\sin^2\varphi\cos\theta\bigr).
\]

Equivalently,

\[
v\cdot R_{e_x}(\theta)v
=1-\sin^2\varphi\,(1-\cos\theta)
=\cos\theta+\cos^2\varphi\,(1-\cos\theta).
\]

**Checks used in the probe.**

- \(\varphi=\pi/2\) (CLI axis in the \(yz\)-plane, default \(v=e_z\)):
  \(v\cdot R_{e_x}(\theta)v=\cos\theta\). For \(\theta\in(0,\pi)\) this is \(\delta=\theta\). Table: \(2.399963\).

- \(\varphi=\pi/4\) (probe \(v=(e_x+e_z)/\sqrt{2}\)):
  \(v\cdot R_{e_x}(\theta)v=\cos^2(\theta/2)\), so
  \(\delta=\arccos\bigl(\cos^2(\theta/2)\bigr)\).
  With \(\theta=2.399963\), \(\cos(\theta/2)\approx0.362375\), \(\cos^2(\theta/2)\approx0.131316\), \(\delta\approx1.439100\).

- \(\varphi=0\) (CLI axis along bake-\(x\)):
  \(R_{e_x}(\theta)v=v\), so \(\delta=0\).

`walk_phase` is the generator angle \(\theta\) on the rotation group. \(\delta\) is that same generator after embedding the axis in \(S^2\). They agree only when \(v\perp e_x\). Off that plane the chord is strictly smaller than \(\theta\) for \(\theta\in(0,\pi)\), because \(\sin^2\varphi<1\) pulls the cosine up toward \(1\).

The formula does not use `n`, \(Z\), or the alkane aliases. It is only the one-step \(S^2\) geometry of `golden_rotate_cli`.
