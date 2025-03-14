---
title: "IDFRecall"
date: "2025-03-15 00:00:00 +0800"
categories: [learning]
tags: [nlp]
math: true
description: IDF recall ranking
pin: true
---

Currently, to obtain the “correct” chunks (paragraphs) for a query $q_i$, different approaches are used. In practice, there are usually two:

1. **Embedding-based search**  
   $R_e(q_i) \mapsto \{d_1, d_2, \ldots, d_k\}$ — here we want to “map” vector representations of queries and paragraphs into a single Euclidean space.

2. **Keyword-based search**  
   $R_k$, using, for example, BM25 or TF-IDF.  
   If you don’t know what these are, [here](https://huggingface.co/blog/xhluca/bm25s) is a resource — read it (and better yet, try it right away!).

3. **Hybrid search**  
   A combination of the first two approaches:

   $$
   R_h = \alpha \times R_e(q_i) + (1 - \alpha) \times R_k(q_i).
   $$

But what if neither approach gives a complete set of results? And you still want to provide context and retrieve keywords? In this case, you have to delve into word ranking.

Meet **IDF-Recall**.

---

![](/docs/sample-hunger-games.png)

**Intuition:** We weight all the words of the query $q_i$ and measure how strongly they cover the keywords of the paragraph (chunk).

For example, consider the query above $q_i = \text{"Какие правила арены голодных игр?"}$ and the provided paragraph $d_1$ (suppose only one document was returned):

> "Правила голодных игр просты. В наказание за прошлый мятеж каждый из двенадцати районов ежегодно отправляет двух трибутов — юношу и девушку. Все трибуты помещаются на арену, где они должны сражаться до последнего выжившего. Арена полностью контролируется Капитолием — чтобы напомнить о власти центра."

- $w_4 = \text{арена (у, ы)}$ — appears twice in the paragraph and is present in the query.  
- $w_3 = \text{трибуты (ов)}$ — appears twice and is in the query.  
- $w_2 = \text{мятеж}$ — appears once.  
- $w_1 = \text{голодные (ых) игры (игр)}$ — appears once.

**Weight** of each word $w_l$ is calculated by the formula:

$$
\frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_l)\bigr)}.
$$

Then the resulting formula for calculating **IDF-Recall** will look like this:

1. Take all the “common” words $w_l \in \{q_i, d_j\}$. In our example, these are $w_1$ and $w_4$ — both appear in the query and are also “rare” words for the paragraph.

2. Take all the “keyword” words of the paragraph (you could simply treat every word as a keyword) — $w_l \in d_j$. In our example, there are four of them: $w_1, w_2, w_3, w_4$.

3. Calculate the “contribution” of each common word to the paragraph’s keywords.

**$R_I$** (short for “IDF recall ranking”) for $(q_i, d_j)$ is defined as:

$$
R_I(q_i, d_j) = 
\frac{\displaystyle \sum_{w_l \,\in\, (q_i \cap d_j)} \frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_l)\bigr)}}
{\displaystyle \sum_{w_l \,\in\, d_j} \frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_l)\bigr)}}.
$$

In our example, it looks like this:

$$
R_I = 
\frac{
\frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_1)\bigr)}
+ 
\frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_4)\bigr)}
}{
\frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_1)\bigr)}
+ 
\frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_2)\bigr)}
+ 
\frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_3)\bigr)}
+ 
\frac{1.0}{\ln\bigl(1 + \mathrm{count}(w_4)\bigr)}
}
\approx
\frac{1}{2}.
$$

Which means that about half of the query’s words “cover” the important words of the document.

❗️Interestingly, if we remove the word “арены” (“arena”) from the query $q_i$, the denominator remains the same, but the numerator decreases, and the final value will be approximately 0.3.

---

In my tests, adding this type of ranking increases **HitRate@k** by an average of **5–10%**. Below is an example of what happens when we use [$E5_{base}$](https://huggingface.co/intfloat/e5-base) as the neural model $R_e$. **$R_{\gamma}$** represents the search accuracy if we choose candidates and then perform additional re-ranking.

Below is the updated table with results from the **polaroids.ai** dataset:

|  Method   | **HitRate@k (Polaroids.ai.Dataset)** |
|:---------:|:-------------------------------------:|
| $R_k$     | 0.49                                  |
| $R_e$     | 0.58                                  |
| $R_h$     | 0.61                                  |
| $R_{\gamma}$ | **0.633**                          |

**Table 1.** *HitRate@k* metric for the $E5_{\text{base}}$ model.

❗️If anything is unclear, feel free to reach out — I’d be happy to answer:

- [*LinkedIn*](https://www.linkedin.com/in/itarlinskiy/)
- [*Telegram*](https://t.me/itarlinskiy/)

