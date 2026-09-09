\# Evaluation Findings



\## What is misleading about my headline number?



The strongest-looking number from the initial experiments is 94% intent accuracy on the 200-example golden set. This number is misleading because the classifier was trained on those same 200 examples before making predictions. It therefore measures memorization and fit on the evaluation data rather than generalization to unseen customer messages.



A leakage-free 5-fold cross-validation gives a more realistic result of 36.0% accuracy and 19.15% macro-F1. The majority-class baseline achieves 34.0% accuracy and 4.61% macro-F1.



The gap between accuracy and macro-F1 is also important. The golden set is highly imbalanced, with delivery\_issue accounting for 68 of the 200 examples while seller\_review has only 2. Accuracy can therefore hide poor performance on minority intents. Macro-F1 gives each intent equal weight and exposes this weakness.



The golden set is also relatively small and contains only two seller\_review examples. Results for rare intents should therefore be treated as directional rather than statistically reliable.



The retrieval result of 84.5% top-1 intent agreement is also not a true retrieval accuracy measure. It is an intent-agreement proxy because the retrieved historical cases were evaluated using classifier-inferred intents rather than hand-labelled intents.



\## Decision Layer Findings



On the 200-example golden set, the decision layer selected auto\_handle for 132 examples and human\_escalation for 68 examples.



There were 2 potential false auto-handles. Both involved multilingual escalation-style messages where the classifier predicted another intent and the retrieval score was sufficient to pass the automation threshold.



There were 14 potential unnecessary escalations. Common causes were the use of words such as "urgent", low retrieval scores just below the threshold, multilingual messages, and customer frustration appearing alongside an otherwise actionable support request.



This suggests that the current decision layer is conservative but relies on simple lexical rules and a hand-selected retrieval threshold. It should not be treated as production-safe without a stronger evaluation set and better uncertainty handling.



\## What I would build next in one week



1\. Expand the golden set from 200 examples to at least 500 examples, with deliberate coverage of minority intents and multilingual messages.



2\. Replace the TF-IDF classifier with a stronger text classifier or embedding-based intent model and compare it against the current baseline.



3\. Improve retrieval using dense embeddings and reranking, while keeping the historical Amazon responses as the grounding source.



4\. Replace keyword-based escalation rules with a calibrated decision model that considers intent confidence, retrieval confidence, escalation language, and message ambiguity together.



5\. Add a dedicated mixed-intent category or multi-label classification for messages containing multiple support issues.



6\. Run a structured human evaluation on generated responses and measure agreement between human reviewers and the LLM judge.



7\. Add explicit safety checks that block automatic replies when the model is uncertain, the retrieved examples disagree, or the response contains unsupported claims.

