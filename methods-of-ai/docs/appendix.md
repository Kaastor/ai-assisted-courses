# Appendix

### Appendix: common nondeterminism sources & fixes (quick list)

* Data order & random sampling → **fix seeds**, `num_workers=0`, `shuffle=True` with seed, disable Python hash randomization.
* Dropout & initialization → seed and log initial weights’ stats.
* Different library versions → log `pip freeze` / `poetry export`.
* Floating‑point non‑associativity → expect tiny metric jitter; compare with tolerances in tests.

---

[1]: https://archive.ics.uci.edu/dataset/2/adult?utm_source=chatgpt.com "Adult - UCI Machine Learning Repository"
[2]: https://github.com/zalandoresearch/fashion-mnist "GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark"
[3]: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection?utm_source=chatgpt.com "SMS Spam Collection"
[4]: https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014?utm_source=chatgpt.com "ElectricityLoadDiagrams20112014"
[5]: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt "files.grouplens.org"
[6]: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html "files.grouplens.org"

