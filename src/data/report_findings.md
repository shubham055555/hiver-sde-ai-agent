\# Evaluation Findings



\## Classification findings



The strongest-looking number from the initial experiments is 94% intent accuracy on the 200-example golden set. This number is misleading because the classifier was trained on those same 200 examples before making predictions. It therefore measures fit on the evaluation data rather than generalization to unseen customer messages.



A leakage-free 5-fold cross-validation gives a more realistic result of 36.0% accuracy and 19.15% macro-F1. The majority-class baseline achieves 34.0% accuracy and 4.61% macro-F1.



The gap between accuracy and macro-F1 is also important. The golden set is highly imbalanced, with `delivery\_issue` accounting for 68 of the 200 examples while `seller\_review` has only 2. Accuracy can therefore hide poor performance on minority intents. Macro-F1 gives each intent equal weight and exposes this weakness.



The golden set is also relatively small and contains only two `seller\_review` examples. Results for rare intents should therefore be treated as directional rather than statistically reliable.



\## Retrieval findings



The retrieval system achieves 84.5% top-1 and 92.0% top-3 intent agreement on the 200-example golden set.



These are not true retrieval accuracy measures. They are intent-agreement proxies because the historical retrieved cases were evaluated using classifier-inferred intents rather than hand-labelled intents.



The main benefit of retrieval is therefore currently demonstrated as candidate-case discovery rather than verified semantic relevance.



\## Response-generation findings



A separate response evaluation set contains 20 examples covering 10 non-escalation intents, with two examples per intent.



The LLM judge produced the following average scores:



| Dimension             |  Average |

| --------------------- | -------: |

| Groundedness          | 5.00 / 5 |

| Helpfulness           | 4.95 / 5 |

| Intent correctness    | 5.00 / 5 |

| No unsupported claims | 5.00 / 5 |

| Overall quality       | 4.95 / 5 |



The overall LLM-judge average across the five dimensions is 4.98 / 5.



However, these scores should not be treated as a definitive response-quality benchmark because the evaluation set contains only 20 examples and the evaluator is itself an LLM.



A human evaluation was performed on the same 20 responses.



Human and LLM-judge overall scores matched exactly on 75% of examples. The mean absolute difference between the two scores was 0.40 points on the 1-5 scale.



Agreement statistics were:



\* Cohen's kappa: 0.231

\* Linear weighted kappa: 0.149

\* Pearson correlation: 0.157



The disagreement shows that the LLM judge is optimistic on some failure cases.



For example, one `other\_non\_support` message received a human overall score of 2/5 while the LLM judge gave it 5/5. The generated response incorrectly treated a sarcastic non-support message as an unresolved support problem.



Another seller-related example received 3/5 from the human evaluator and 5/5 from the LLM judge because the response requested account-related information that was not clearly necessary from the customer message.



Therefore, the human evaluation is more useful for identifying concrete response-quality failures, while the LLM judge is treated as a scalable but imperfect evaluator.



\## Decision layer findings



On the 200-example golden set, the decision layer selected `auto\_handle` for 132 examples and `human\_escalation` for 68 examples.



There were 2 potential false auto-handles. Both involved multilingual escalation-style messages where the classifier predicted another intent and the retrieval score was sufficient to pass the automation threshold.



There were 14 potential unnecessary escalations. Common causes were the use of words such as "urgent", low retrieval scores just below the threshold, multilingual messages, and customer frustration appearing alongside an otherwise actionable support request.



This suggests that the current decision layer is conservative but relies on simple lexical rules and a hand-selected retrieval threshold. It should not be treated as production-safe without a stronger evaluation set and better uncertainty handling.



\## Top failure modes



\### 1. Multilingual and short messages



Short Spanish and Portuguese messages can be classified incorrectly because the classifier is trained mainly on English-language support text.



One Spanish escalation-style message was predicted as `product\_information` instead of `support\_escalation`.



\### 2. Delivery issues mixed with escalation language



Customers can describe a normal delivery problem while also using words such as "urgent", "complaint", or "useless customer care".



The keyword-based decision layer can escalate these cases even when the underlying support issue may be actionable.



\### 3. Retrieval-context mismatch



A high retrieval score does not always guarantee that the retrieved example is actually useful for response generation.



Some messages received highly similar lexical matches that were still poor resolution examples.



\### 4. Mixed-intent messages



Some customer messages contain multiple issues, such as delivery combined with payment or legal language.



The current single-label taxonomy forces the classifier to choose one primary intent.



\### 5. Casual and sarcastic messages



Casual, thank-you, sarcastic, or non-support messages can be converted into generic support requests by the response generator.



One `other\_non\_support` example received a human response-quality score of only 2/5 because the generated reply asked for more details about an issue that the customer had not actually raised.



\## What is misleading about my headline number?



The most misleading headline number is the 94% intent accuracy obtained when training and evaluating on the same 200 golden examples.



That result is not a valid estimate of generalization because the evaluation examples were also used for training.



A leakage-free five-fold evaluation reduces performance to 36.0% accuracy and 19.15% macro-F1.



The 84.5% retrieval result also needs qualification because it is an intent-agreement proxy rather than manually verified retrieval accuracy.



Finally, the 4.95/5 LLM-judge overall response score should not be presented without the human-agreement context. The judge had only 75% exact agreement with human overall ratings and a Cohen's kappa of 0.231.



The headline numbers therefore need to be presented together with their evaluation limitations.



\## What I would build next in one week



1\. Expand the golden set from 200 examples to at least 500-1,000 examples, with deliberate coverage of minority intents and multilingual messages.



2\. Replace the TF-IDF classifier with a stronger text classifier or embedding-based intent model and compare it against the current baseline.



3\. Improve retrieval using dense embeddings and reranking, while keeping historical Amazon responses as the grounding source.



4\. Replace keyword-based escalation rules with a calibrated decision model that considers intent confidence, retrieval confidence, escalation language, and message ambiguity together.



5\. Add multi-label classification or a dedicated mixed-intent treatment for messages containing multiple support issues.



6\. Use multiple human reviewers for the response evaluation set and measure inter-rater agreement before using the set as a stronger benchmark.



7\. Add explicit safety checks that block automatic replies when the model is uncertain, retrieved examples disagree, or the generated response contains unsupported claims.



