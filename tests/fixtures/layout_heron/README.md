# Heron benchmark fixture

`benchmark_gradient.png` is a repository-owned deterministic RGB fixture with size 401×534. It is
generated from zero-based integer pixel coordinates `(x, y)` as:

```text
R = (3x + 5y + 17) mod 256
G = (11x + 7y + 29) mod 256
B = (13x + 19y + 43) mod 256
```

The PNG SHA-256 is `7b15f67a2069a7ef733b44a3e44a9a04ee4fc9b07d9ca69c683b6f3e2ae61b58`.
It is used for prepared-image Metal benchmarks; PDF rendering and page crop are outside that timing
boundary.
