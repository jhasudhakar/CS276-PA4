# CS276-PA3

In this programming assignment we use [learning to rank](http://research.microsoft.com/en-us/people/hangli/li-acl-ijcnlp-2009-tutorial.pdf) techniques to rank documents. Specifically, we implemented:

1. Pointwise approach with linear regression
2. Pointwise approach with SVR
3. Pairwise approach with SVM with linear and RBF kernel

For details please read `report.pdf`.

## Thoughts

Although machine learning is powerful and promising in information retrieval, to get it right requires much effort as well as experience. You need to find good features and avoid gotchas (e.g. feature preprocessing, standardization).

Moreover, in pairwise approach, high SVM predication accuracy doesn't necessarily indicate high NDCG score. As always, automation is your friend - systematic grid search is better and ad-hoc search.
