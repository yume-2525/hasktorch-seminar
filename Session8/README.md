# Improvement of the Session7 Task

## Creation of a New Vocabulary Dictionary
An attempt was made to improve the self-built embedding, but due to a lack of memory, it was decided to use an existing pre-trained model.
300-dimensional word vector data was obtained from the [GloVe](https://nlp.stanford.edu/projects/glove/) website.
Since the data size was too large, not all of it was used. The top 10,000 words were extracted and used.
Additionally, to handle out-of-vocabulary (OOV) words, a process was implemented to append one zero vector to the end of the embedding.

## Increasing the Number of Dimensions
The new vocabulary dictionary has 300  dimensions of the word vectors .
Furthermore, to enhance the expressiveness of the model, the number of dimensions of the RNN's hidden state was set to 256.
However, due to memory constraints, the number of training iterations was set to 300.

## Improvement of the Evaluation Method
To understand the model's performance in more detail, the evaluation were improved as follows:
+ Added the measurement of accuracy, F1 score, and out-of-vocabulary (OOV) rate.
+ Displayed a Confusion Matrix to show the bias in the prediction results.

## Results
Conditions:  
 + Number of iterations: 300
 + Learning rate: 0.0001
 + Batch size: 16
 + Number of word vector dimensions: 300
 + Hidden state layer: 256  

![](./result_graph/reviewRNN-emb-1.png)

```
===== 評価結果 =====
【正解率 (Accuracy)】: 13.0199995 %
【Macro F1】   : 8.095141e-2
【Weighted F1】: 4.7293276e-2
【未知語率】  : 20723 / 228241 (9.079437 %)
【未知語彙率】  : 5824 / 10000 (58.24 %)

【Confusion Matrix】
      Prediction1 Prediction2 Prediction3 Prediction4 Prediction5
Answer1: [0,169,156,1184,0]
Answer2: [0,55,74,375,0]
Answer3: [0,88,77,696,0]
Answer4: [0,179,101,1170,0]
Answer5: [1,775,375,4525,0]
```

+ Predicted values almost never became "1" or "5," and the results were extremely biased, with the large majority falling into "4."
+ Accuracy still remains low.
+ The F1 score, which indicates the balance between precision and recall, also showed very low values.
+ On the other hand, due to the improvement in preprocessing, the OOV rate decreased to 9%, showing a significant improvement. Although the OOV vocabulary rate is still high, it is thought that these are words with low frequency of occurrence.

## Discussion
Last time, a hypothesis was formed that predictions might bias toward "4" because "4" is close to the average value of the entire data.
This time, although the foundational reading comprehension improved by increasing the number of dimensions and improving the initial values of the embedding, the accuracy still remained low.

The one of  causes of this is considered to be the use of `mseLoss` (Mean Squared Error) as the loss function. Since `mseLoss` evaluates errors by squaring them, the penalty becomes extremely large when making an extreme prediction (such as 1 or 5) and missing by a wide margin. Therefore, it is highly possible that the model learned to output a "safe average value (around 4)" in an attempt to minimize the penalty.

By next time, I would like to change the task from regression to multi-class classification, and compare and verify the results using `nllLoss` (Negative Log-Likelihood Loss) as the loss function.


```
【正解】: 5 | 【予測】: 4 (3.7983966) | OOV: 4/27 | 【本文】: I have used Quicken for many years.  Although I think it is a rip to have to upgrad every 3 years, I can't fault this product.
【正解】: 1 | 【予測】: 4 (3.942133) | OOV: 0/12 | 【本文】: It does not work. I have tried it a number of times.
【正解】: 5 | 【予測】: 4 (4.1485863) | OOV: 2/12 | 【本文】: Have used TurboTax products since the 1990s.  Still pleased with them.
【正解】: 5 | 【予測】: 4 (3.5118241) | OOV: 0/4 | 【本文】: Actual news without spin
【正解】: 5 | 【予測】: 4 (3.6021247) | OOV: 0/2 | 【本文】: Nice channel
【正解】: 5 | 【予測】: 4 (3.6070457) | OOV: 0/2 | 【本文】: GREAT CHANNEL.
【正解】: 1 | 【予測】: 4 (3.7861226) | OOV: 1/12 | 【本文】: I have played a lot of solitaire and this is the worst.
【正解】: 1 | 【予測】: 3 (3.377277) | OOV: 0/1 | 【本文】: Boring.
【正解】: 5 | 【予測】: 4 (3.8027816) | OOV: 0/1 | 【本文】: Fun!
【正解】: 4 | 【予測】: 4 (3.739414) | OOV: 3/33 | 【本文】: There are some glitches on the transfer of info to state return,  Even though every 1099 entered shows who it was for, the proper allocation is not done on the state return.
【正解】: 3 | 【予測】: 4 (3.7764745) | OOV: 2/33 | 【本文】: The listing for this product would make one believe the price covered federal and state tax preparation.  Imagine my surprise to learn I had to pay an additional $39.99 for state program.
【正解】: 4 | 【予測】: 4 (3.8463495) | OOV: 9/53 | 【本文】: Got this free for a year with my new iPhone purchase.  There isn't a ton of content on this platform but what they do have is high quality.  Watch the movie "Greyhound" with Tom hanks and currently watching a series with Jennifer Aniston called "The Morning Show" both are AppleTV+ only.
【正解】: 5 | 【予測】: 4 (3.6512206) | OOV: 3/16 | 【本文】: works well it's norton much cheaper than auto renewal oh yeah and it's comfortable check stars
【正解】: 1 | 【予測】: 2 (1.875376) | OOV: 3/43 | 【本文】: The game no longer works on Amazon 10 tablet. It never gets past the load screen. Was a fun game when I could play it. Don't know if it can be after an update or not. Have tried to uninstall and install again.nope.
【正解】: 4 | 【予測】: 4 (3.5654972) | OOV: 0/14 | 【本文】: Simply easy and accurate, just what I need at the tip of my fingers!
【正解】: 2 | 【予測】: 4 (4.082902) | OOV: 6/25 | 【本文】: Playing with the car?  I just needed a dependable alarm.  This wasn't it.  I deleted it.  Probably should have returned it.
【正解】: 1 | 【予測】: 2 (1.8728392) | OOV: 1/2 | 【本文】: seemed babyish.
【正解】: 5 | 【予測】: 4 (3.9073205) | OOV: 0/7 | 【本文】: good price and was delivered very quickly
【正解】: 5 | 【予測】: 4 (4.130128) | OOV: 1/10 | 【本文】: great price and quick delivery. mcAfee always works for me
【正解】: 1 | 【予測】: 3 (3.2158663) | OOV: 0/1 | 【本文】: Garbage
【正解】: 1 | 【予測】: 2 (1.9054455) | OOV: 3/43 | 【本文】: I tried the free and payed for a month. Than l had it cancelled. Was supposed to be coming out of my Amazon credit card. Looking today it's still coming out of my 5th 3rd credit card. How do I cancel this ?
【正解】: 5 | 【予測】: 4 (3.503462) | OOV: 0/4 | 【本文】: Great product no complaints
【正解】: 5 | 【予測】: 4 (3.7609925) | OOV: 0/14 | 【本文】: Love game for all ages building your own zoo as you pass match levels
【正解】: 5 | 【予測】: 4 (4.0169606) | OOV: 1/7 | 【本文】: Useing on my fire tablet working great
【正解】: 4 | 【予測】: 4 (4.0843325) | OOV: 0/3 | 【本文】: Relaxing and fun
【正解】: 4 | 【予測】: 4 (3.6982965) | OOV: 0/11 | 【本文】: Very challenging game for young and older adults! Enjoy using brain.
【正解】: 5 | 【予測】: 4 (3.6035903) | OOV: 4/39 | 【本文】: You will 💘 creating fish bowls, while using your brain to. You will 💘 your fishies and they love you back. Decorate the way you want you can move things to other fish bowls stoore items just great enjoyment!!!
【正解】: 4 | 【予測】: 3 (3.4031203) | OOV: 7/40 | 【本文】: Love game good for the brain.<br /><br />  very challenging their hardest game, for any age that's what is important most of all. And the enjoyment of helping is learning . And teaches patience is worth it.and love gardening.
【正解】: 5 | 【予測】: 3 (3.3400793) | OOV: 0/2 | 【本文】: Love spinning.
【正解】: 4 | 【予測】: 4 (4.1442385) | OOV: 0/2 | 【本文】: Enjoyable game!
【正解】: 5 | 【予測】: 4 (4.0818167) | OOV: 0/6 | 【本文】: Very engaging, enjoying it a lot.
【正解】: 4 | 【予測】: 4 (3.8817475) | OOV: 3/49 | 【本文】: I have used several products like TaxCut. I think TaxCut is the most user friendly. This is the eight year I have used TaxCut and feel the results are good. Using the same software each year saves you time because you can roll last years information into this years.
【正解】: 5 | 【予測】: 2 (1.6083566) | OOV: 1/1 | 【本文】: worksgreat
【正解】: 3 | 【予測】: 4 (4.092625) | OOV: 1/12 | 【本文】: I have SlingTv and it has almost all the channels I need
【正解】: 5 | 【予測】: 4 (4.00581) | OOV: 0/2 | 【本文】: Good stuff
【正解】: 2 | 【予測】: 4 (3.6054661) | OOV: 0/3 | 【本文】: Zero production values.
【正解】: 4 | 【予測】: 4 (4.0794573) | OOV: 0/5 | 【本文】: I like it works good
【正解】: 3 | 【予測】: 4 (3.8495328) | OOV: 1/14 | 【本文】: Just got. It, don't know if I like it yet, lots to look at.
【正解】: 5 | 【予測】: 4 (3.938607) | OOV: 1/13 | 【本文】: I like it but can't get google chrome on my amazon fire, why?
【正解】: 3 | 【予測】: 3 (3.3947337) | OOV: 1/7 | 【本文】: Fun game,challenging, but no close or exit
【正解】: 5 | 【予測】: 3 (3.4037058) | OOV: 1/3 | 【本文】: Luv u tube
【正解】: 1 | 【予測】: 2 (1.6083566) | OOV: 1/1 | 【本文】: Uggg
【正解】: 5 | 【予測】: 2 (1.6083566) | OOV: 1/1 | 【本文】: 👍
【正解】: 5 | 【予測】: 4 (3.7853541) | OOV: 2/13 | 【本文】: Did every thing perfect.  Easy Easy Easy To Install.  Great Buy
【正解】: 5 | 【予測】: 4 (3.927301) | OOV: 7/32 | 【本文】: Great Deal...I have been using AVG for 10+ years.  Their products get better every year.  Been enjoying &#34;virus&#34; and &#34;malware&#34; free computer operation for as long as I can remember.
【正解】: 4 | 【予測】: 2 (1.6083566) | OOV: 1/1 | 【本文】: Yellowstone
【正解】: 5 | 【予測】: 4 (3.7358248) | OOV: 0/1 | 【本文】: Great!
【正解】: 3 | 【予測】: 3 (3.4759493) | OOV: 0/1 | 【本文】: OK
【正解】: 5 | 【予測】: 4 (3.8575728) | OOV: 0/2 | 【本文】: Great quality.
【正解】: 3 | 【予測】: 4 (3.7606955) | OOV: 1/7 | 【本文】: It shuts down and it goes slow
```