Integer Factorization of an integer Gram Matrix
===============================================

The general problem is: given a positive definite $n \times n$ integer
symmetric matrix $G$, find an $n \times m$ integer matrix $U$ for some
$m \ge n$, such that $U U^T = G$, or prove that no such matrix exists.

This is actually the classical problem of *representing one quadratic
form by another*.  In abstract terms we are given $Q$ a quadratic map:
1) $Q: \mathbb{R}^n \rightarrow \mathbb{R}_{\ge 0}$.
2) $F(x,y) := \frac 12 (Q(x+y) - Q(x) - Q(y))$ is a bilinear map.
3) $Q(x) \in \mathbb{Z}$ for all $x \in \mathbb{Z}^n$.
4) $Q(x) > 0$ if $x \ne 0$.

Given quadratic maps $Q: \mathbb{R}^n \rightarrow \mathbb{R}$ and
$R: \mathbb{R}^m \rightarrow \mathbb{R}$, with $m > n$ is there an
integral $m \times n$ matrix $U$, such that $Q(x) = R(U x)$ for all $x
\in \mathbb{R}^n$?
