# 1. Improvement of the Session7 Task
Based on the previous feedback, several settings were changed to aim for improvement.

### 1.1 Experimental Conditions and List of Results
The data preprocessing conditions and the model (RNN/LSTM) were changed, and the trends in Accuracy and F1 score were verified.

This week, the code was rewritten by adding to the conditions of last week.
Conditions:  
 + Number of iterations: 100
 + Learning rate: 0.0001
 + Batch size: 16
 + Number of word vector dimensions: 300
 + Hidden state layer: 256  

First, the results from last week are as follows:

 ```
===== Evaluation Results =====
[Accuracy]                  : 13.0199995 %
[Macro F1]                  : 8.095141e-2
[Weighted F1]               : 4.7293276e-2
[OOV Rate]                  : 20723 / 228241 (9.079437 %)
[OOV Vocabulary Rate]       : 5824 / 10000 (58.24 %)

[Confusion Matrix]
Prediction1 Prediction2 Prediction3 Prediction4 Prediction5
Answer1: [0, 169, 156, 1184, 0]
Answer2: [0, 55, 74, 375, 0]
Answer3: [0, 88, 77, 696, 0]
Answer4: [0, 179, 101, 1170, 0]
Answer5: [1, 775, 375, 4525, 0]
 ```

#### Exp1: Sentences with 3 words or fewer were excluded only from the training data.
```
[Accuracy]                  : 13.13 %
[Macro F1 Score]            : 7.1647406e-2
[Weighted F1 Score]         : 4.3403126e-2
[OOV Rate]                  : 20723 / 228241 (9.079437 %)
[OOV Vocabulary Rate]       : 5824 / 10000 (58.24 %)

[Confusion Matrix]
Predicted1 Predicted2 Predicted3 Predicted4 Predicted5
Actual 1: [0, 169, 101, 1239, 0]
Actual 2: [0, 55, 34, 415, 0]
Actual 3: [0, 88, 28, 745, 0]
Actual 4: [0, 179, 41, 1230, 0]
Actual 5: [1, 775, 174, 4726, 0]
```

#### Exp2: Sentences with 3 words or fewer were excluded from both the training data and test data.
```
===== Evaluation Results =====
[Accuracy]                  : 16.495619 %
[Macro F1 Score]            : 0.10356808
[Weighted F1 Score]         : 9.762956e-2
[OOV Rate]                  : 20310 / 224294 (9.055079 %)
[OOV Vocabulary Rate]       : 5451 / 10000 (54.51 %)

[Confusion Matrix]
Predicted1 Predicted2 Predicted3 Predicted4 Predicted5
Actual 1: [148, 0, 79, 978, 25]
Actual 2: [49, 0, 41, 333, 6]
Actual 3: [79, 0, 39, 581, 10]
Actual 4: [166, 0, 62, 1008, 29]
Actual 5: [613, 0, 222, 3399, 123]
```

#### Exp3: Sentences with 3 words or fewer were excluded from both the training data and test data. In addition, the training data was leveled so that the number of sentences for each score became the same before training.
```
===== Evaluation Results =====
[Accuracy]                  : 7.0463076 %
[Macro F1 Score]            : 5.201487e-2
[Weighted F1 Score]         : 2.9692741e-2
[OOV Rate]                  : 20310 / 224294 (9.055079 %)
[OOV Vocabulary Rate]       : 5451 / 10000 (54.51 %)

[Confusion Matrix]
Predicted1 Predicted2 Predicted3 Predicted4 Predicted5
Actual 1: [209, 1021, 0, 0, 0]
Actual 2: [76, 353, 0, 0, 0]
Actual 3: [113, 595, 1, 0, 0]
Actual 4: [223, 1042, 0, 0, 0]
Actual 5: [830, 3526, 1, 0, 0]
```

#### Exp4: Sentences with 3 words or fewer were excluded from both the training data and test data. In addition, from the training data and test data, those with an OOV rate of 20% or less were adopted
```
===== Evaluation Results =====
[Accuracy]                  : 14.623918 %
[Macro F1 Score]            : 8.0306746e-2
[Weighted F1 Score]         : 5.418222e-2
[OOV Rate]                  : 17655 / 214197 (8.242413 %)
[OOV Vocabulary Rate]       : 4853 / 10000 (48.53 %)

[Confusion Matrix]
Predicted1 Predicted2 Predicted3 Predicted4 Predicted5
Actual 1: [10, 105, 43, 952, 0]
Actual 2: [4, 35, 25, 333, 0]
Actual 3: [8, 61, 25, 575, 0]
Actual 4: [19, 114, 50, 1011, 0]
Actual 5: [46, 392, 161, 3423, 0]
```

#### Exp6: Sentences with 3 words or fewer were excluded from both the training data and test data. In addition, from the training data and test data, those with an OOV rate of 0% were adopted
```
===== 評価結果 =====
【正解率 (Accuracy)】: 14.80898 %
【Macro F1スコア】   : 6.5599576e-2
【Weighted F1スコア】: 4.6019718e-2
【未知語率】  : 0 / 29126 (0.0 %)
【未知語彙率】  : 0 / 10000 (0.0 %)

【Confusion Matrix】
      予測1 予測2 予測3 予測4 予測5
正解1: [0,0,23,260,0]
正解2: [0,0,9,75,0]
正解3: [0,0,12,190,0]
正解4: [0,0,30,364,0]
正解5: [0,0,96,1480,0]
```

#### Exp8: Sentences with 3 words or fewer were excluded from both the training data and test data. An LSTM was used.
```
===== Evaluation Results =====
[Accuracy]                  : 9.036295 %
[Macro F1 Score]            : 6.4841114e-2
[Weighted F1 Score]         : 3.221859e-2
[OOV Rate]                  : 20310 / 224294 (9.055079 %)
[OOV Vocabulary Rate]       : 5451 / 10000 (54.51 %)

[Confusion Matrix]
Predicted1 Predicted2 Predicted3 Predicted4 Predicted5
Actual 1: [1, 152, 1033, 44, 0]
Actual 2: [0, 54, 360, 15, 0]
Actual 3: [0, 80, 597, 32, 0]
Actual 4: [3, 165, 1027, 70, 0]
Actual 5: [28, 608, 3493, 228, 0]
```

| Exp | 条件（前処理・モデル） | Accuracy | Macro F1 | 備考・特徴 |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 先週の結果 | 13.02% | 0.081 | 予測が「4」に偏る |
| **1** | 訓練データのみ3語以下除外 | 13.13% | 0.071 | 予測が「2, 4, 5」に偏る |
| **2** | 訓練＆テスト共に3語以下除外 | **16.49%** | **0.103** | 予測の幅が広がり、本実験の最高精度 |
| **3** | Exp2 ＋ 訓練データスコア均等化 | 7.04% | 0.052 | 「1, 2」の予測に極端に偏る |
| **4** | Exp2 ＋ 未知語率20%以下（訓練のみ） | 14.62% | 0.080 | 予測が再び「4, 5」の中央付近に集中 |
| **6** | 訓練＆テスト共に未知語率0% | 15.28% | 0.056 | データ激減（訓練819件）。分散は小さい |
| **8** | Exp2の条件 ＋ **LSTMモデル**に変更 | 9.03% | 0.064 | 長期記憶を導入したが予測は4・5に偏る |

### 1.2 Discussion and Issues
In any case, good results could not be obtained. Running it on this computer, the memory shortage limits the number of iterations to around 100 times, which was considered to be one of the causes. Also, the fact that the amount of data has become considerably small compared to the whole due to extraction is also considered to be one of the causes.

---

## 2. Word Embedding Meaning Composition (Using GloVe)
Session6 task4 was conducted.
> **Session6 task4**
> (advanced only for word2vec) It is known that vectors learned with word2vec can be used for calculating the meaning composition (e.g., “king” – “male” + “female” = “queen” ). Check if your model predicts this.

### 2.1 Implementation
  1. Loading the vocabulary dictionary
  2. Loading the word vectors
  3. Obtaining the vectors of the 3 target words
  4. Operation of the vectors
  5. Calculating the similarity with all word vectors
  6. Sorting in descending order of similarity
  7. Displaying the top 10 items

### 2.2 Results

* `king - man + woman` ⇒ **1st: queen (0.884)**, 3rd: prince (0.802), 4th: daughter
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "queen"           | Similarity: 0.88415766
Word: "king"            | Similarity: 0.8716328
Word: "prince"          | Similarity: 0.8021438
Word: "daughter"                | Similarity: 0.8008097
Word: "princess"                | Similarity: 0.80009305
Word: "mother"          | Similarity: 0.79811996
Word: "woman"           | Similarity: 0.79135954
Word: "lady"            | Similarity: 0.7825714
Word: "wife"            | Similarity: 0.7778128
Word: "sister"          | Similarity: 0.77610946
```

</details>

* `sister - woman + man` ⇒ **1st: brother (0.916)**, 2nd: uncle (0.880), 3rd: father
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "brother"         | Similarity: 0.91646194
Word: "uncle"           | Similarity: 0.8803776
Word: "father"          | Similarity: 0.8729759
Word: "sister"          | Similarity: 0.8646913
Word: "knew"            | Similarity: 0.8554758
Word: "friend"          | Similarity: 0.85090125
Word: "cousin"          | Similarity: 0.84914327
Word: "dad"             | Similarity: 0.84631556
Word: "brothers"                | Similarity: 0.84387493
Word: "son"             | Similarity: 0.8382679
```

</details>



* `went - go + eat` ⇒ **1st: ate (0.942)**, 3rd: eaten (0.913), 4th: eating
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "ate"             | Similarity: 0.9428296
Word: "eat"             | Similarity: 0.92971766
Word: "eaten"           | Similarity: 0.91367066
Word: "eating"          | Similarity: 0.9032604
Word: "meal"            | Similarity: 0.8117261
Word: "eats"            | Similarity: 0.81111574
Word: "food"            | Similarity: 0.80138713
Word: "hungry"          | Similarity: 0.7760783
Word: "dinner"          | Similarity: 0.77253443
Word: "lunch"           | Similarity: 0.7725326
```

</details>

* `dogs - dog + cat` ⇒ **1st: cats (0.951)**, 2nd: cat (0.925)
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "cats"            | Similarity: 0.95120364
Word: "cat"             | Similarity: 0.9251083
Word: "dogs"            | Similarity: 0.8935337
Word: "dog"             | Similarity: 0.82261205
Word: "animals"         | Similarity: 0.82022715
Word: "pets"            | Similarity: 0.8151135
Word: "pet"             | Similarity: 0.7794855
Word: "animal"          | Similarity: 0.77460843
Word: "rats"            | Similarity: 0.74771565
Word: "rabbit"          | Similarity: 0.73773175
```

</details>




* `happy - good + bad` ⇒ **3rd: sad (0.894)**, 4th: sorry (0.881)
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "happy"           | Similarity: 0.9071727
Word: "bad"             | Similarity: 0.9045669
Word: "sad"             | Similarity: 0.89419097
Word: "sorry"           | Similarity: 0.88198453
Word: "everyone"                | Similarity: 0.8794676
Word: "anymore"         | Similarity: 0.87941915
Word: "guess"           | Similarity: 0.87676996
Word: "me"              | Similarity: 0.87630916
Word: "okay"            | Similarity: 0.875247
Word: "remember"                | Similarity: 0.87433815
```

</details>

* `winter - cold + hot` ⇒ **1st: summer (0.853)**
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "summer"          | Similarity: 0.85348064
Word: "winter"          | Similarity: 0.84981203
Word: "spring"          | Similarity: 0.8045437
Word: "hot"             | Similarity: 0.7927544
Word: "autumn"          | Similarity: 0.7914055
Word: "weekend"         | Similarity: 0.7624954
Word: "hottest"         | Similarity: 0.7499854
Word: "holiday"         | Similarity: 0.74897814
Word: "enjoy"           | Similarity: 0.7474966
Word: "snow"            | Similarity: 0.7403409
```

</details>




* `teacher - school + hospital` ⇒ **2nd: nurse (0.836)**, 3rd: patient (0.805), 4th: doctor
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "hospital"                | Similarity: 0.8907189
Word: "nurse"           | Similarity: 0.8366908
Word: "patient"         | Similarity: 0.80597514
Word: "doctor"          | Similarity: 0.79804164
Word: "medical"         | Similarity: 0.79562247
Word: "physician"               | Similarity: 0.78262234
Word: "clinic"          | Similarity: 0.78236663
Word: "doctors"         | Similarity: 0.77687895
Word: "nurses"          | Similarity: 0.7760856
Word: "hospitals"               | Similarity: 0.7520248
```

</details>

* `breakfast - morning + evening` ⇒ **2nd: dinner (0.894)**
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "breakfast"               | Similarity: 0.94080365
Word: "dinner"          | Similarity: 0.8942672
Word: "lunch"           | Similarity: 0.87023544
Word: "meal"            | Similarity: 0.826232
Word: "guests"          | Similarity: 0.8132275
Word: "meals"           | Similarity: 0.81178284
Word: "dining"          | Similarity: 0.79969287
Word: "buffet"          | Similarity: 0.79237455
Word: "restaurant"              | Similarity: 0.79211414
```

</details>

* `family - human + building` ⇒ **1st: house (0.815)**, 3rd: home (0.794)
<details>
<summary>Details</summary>

```
===== Top 10 Results =====
Word: "house"           | Similarity: 0.8156739
Word: "building"                | Similarity: 0.8031926
Word: "home"            | Similarity: 0.7948838
Word: "houses"          | Similarity: 0.7805037
Word: "family"          | Similarity: 0.7673507
Word: "apartment"               | Similarity: 0.7600019
Word: "homes"           | Similarity: 0.7556403
Word: "built"           | Similarity: 0.7478799
Word: "apartments"              | Similarity: 0.7343205
Word: "residence"               | Similarity: 0.73393977
```

</details>