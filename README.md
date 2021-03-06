# Decision Trees


### 2. MONK Datasets
#### Assignment 0:
> Each one of the datasets has properties which makes them hard to learn. Motivate which of the three problems is most diﬃcult for a decision tree algorithm to learn.


* Dataset 2 is the most difficult to learn because the value of an attribute must be the same as another value of another attribute which make it hard to split on a single attribute.

---

### 3. Entropy
#### Assignment 1:
> The ﬁle `dtree.py` deﬁnes a function entropy which calculates the entropy of a dataset. Import this ﬁle along with the monks datasets and use it to calculate the entropy of the training datasets.


| Dataset | Entropy            |
|---------|--------------------|
| MONK-1  | 1.0                |
| MONK-2  | 0.957117428264771  |
| MONK-3  | 0.9998061328047111 |

#### Assignment 2: 
> Explain entropy for a uniform distribution and a non-uniform distribution, present some example distributions with high and low entropy.

Entropy is a measure of unpredictability of the state.
* In a **uniform distribution** which means that the different outcomes are equally probable Entropy grows logaritmic in relation with the number of outcomes. For example a perfect die or a fair coin.
* In a **non-uniform distribution** which means that one or more outcomes are more probable to occur than the others is always less than or equal to log<sub>2</sub>(*n*). For example a unbalanced die or an unfair coin.
---

### 4. Information Gain
#### Assignment 3:
> Use the function `averageGain` (deﬁned in `dtree.py`) to calculate the expected information gain corresponding to each of the six attributes. Note that the attributes are represented as instances of the class Attribute (deﬁned in `monkdata.py`) which you can access via `m.attributes[0]`, ..., `m.attributes[5]`. Based on the results, which attribute should be used for splitting the examples at the root node?

| Dataset |     a1     |     a2     |     a3     |     a4     |     a5     |     a6     |
|---------|------------|------------|------------|------------|------------|------------|
|  MONK-1 | 0.07527255 | 0.00583843 | 0.00470757 | 0.0263117  | **0.28703075** | 0.00075786 |
|  MONK-2 | 0.00375618 | 0.0024585  | 0.00105615 | 0.01566425 | **0.01727718** | 0.00624762 |
|  MONK-3 | 0.00712087 | **0.29373617** | 0.00083111 | 0.00289182 | 0.25591172 | 0.00707703 |

**a5** performs well on both MONK-1 and MONK-2 and **a2** is better for MONK-3.

#### Assignment 4:
> For splitting we choose the attribute that maximizes the information gain, Eq.3. Looking at Eq.3 how does the entropy of
the subsets, *S<sub>k</sub>*, look like when the information gain is maximized? How can we motivate using the information gain as a heuristic for picking an attribute for splitting? Think about reduction in entropy after the split and what the entropy implies.

The entropy of the subsets is minimized when the information gain is maximized.

Minimized entropy for a subset is optimal since low entropy means it's leaning towards a decision

---

### 5. Building Decision Trees
#### Assignment 5:
> Build the full decision trees for all three Monk datasets using `buildTree`. Then, use the function `check` to measure the performance of the decision tree on both the training and test datasets. <br> For example to built a tree for `monk1` and compute the performance on the test data you could use
```python
import monkdata as m
import dtree as d
t = d.buildTree(m.monk1, m.attributes)
print(d.check(t, m.monk1test))
```
> Compute the train and test set errors for the three Monk datasets for the full trees. Were your assumptions about the datasets correct? Explain the results you get for the training and test datasets.

| Dataset | Error (train) |     Error (test)    |
|---------|---------------|---------------------|
|  MONK-1 |     1.0       | 0.8287037037037037  |
|  MONK-2 |     1.0       | 0.6921296296296297  |
|  MONK-3 |     1.0       | 0.9444444444444444  |

The assumption is correct. Monk-2 has the most errors and thus is the most difficult to learn.

---

### 6. Pruning
#### Assignment 6:
> Explain pruning from a bias variance trade-off perspective.

Simpler model means higher bias and lower variance. pruning is used to lower variance.
Pruning a tree means removing useless overfitting nodes, and by doing that we decrease the variande and increase the bias.

#### Assignment 7:
> Evaluate the eﬀect pruning has on the test error for the `monk1` and `monk3` datasets, in particular determine the optimal partition into training and pruning by optimizing the parameter `fraction`. Plot the classiﬁcation error on the test sets as a function of the parameter `fraction` <span>&#8712;</span> {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}. Reasonable statistics includes mean and a measure of the spread. Do remember to print axes labels, legends and data points as you will not pass without them.



<p align="center"><img src="https://github.com/sork01/dd2421dtree/blob/master/1.png"></p>
