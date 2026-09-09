\# Decision Log



\## 1. Selected AmazonHelp as the primary brand



AmazonHelp had the largest number of brand-to-customer interactions among the candidate brands in the dataset. This provided a larger historical support corpus for both retrieval and response grounding.



\## 2. Reconstructed conversations instead of treating tweets independently



The dataset contains response and parent tweet IDs, so conversations were reconstructed into threads. This preserves the context in which Amazon historically handled customer issues.



\## 3. Removed conversations longer than 20 messages from the retrieval corpus



Very long conversations can contain multiple unrelated issues and make retrieval less precise. Removing them creates a cleaner corpus for matching individual support requests.



\## 4. Defined intents around customer intent rather than product category



The taxonomy focuses on what the customer wants help with, such as delivery, refunds, billing, account access, or technical problems. This makes the classifier more useful for deciding how a support request should be handled.



\## 5. Used a small manually labelled golden set



A 200-example hand-labelled set was created instead of relying entirely on automatically generated labels. This provides a human reference set for evaluating the classifier and decision layer.



\## 6. Used a majority-class baseline



The majority baseline establishes how much performance comes simply from predicting the most frequent intent. This prevents a more complex classifier from appearing useful without comparison.



\## 7. Used TF-IDF with unigrams and bigrams



TF-IDF was chosen as a transparent and inexpensive baseline. Bigrams help capture short phrases such as delivery-related and refund-related expressions while keeping the model easy to inspect.



\## 8. Used class-balanced logistic regression



The golden set is highly imbalanced. Class weighting gives minority intents more influence during training rather than allowing the largest class to dominate the classifier.



\## 9. Used 5-fold cross-validation for the classifier result



The initial 94% result came from evaluating on examples used during training and was therefore rejected as a headline metric. Five-fold cross-validation provides a leakage-free estimate on held-out examples.



\## 10. Used historical customer-response pairs for retrieval



The response generator is grounded in actual Amazon support behaviour from the selected brand instead of relying only on general language-model knowledge.



\## 11. Used TF-IDF cosine similarity for the first retrieval system



A sparse lexical retriever was chosen because it is simple, fast, interpretable, and provides a useful baseline before introducing more complex embedding-based retrieval.



\## 12. Treated retrieval intent agreement as a proxy rather than retrieval accuracy



The retrieved historical cases were not independently hand-labelled for intent. Their intent was inferred using the classifier, so the resulting 84.5% top-1 number is reported as an intent-agreement proxy rather than true retrieval accuracy.



\## 13. Added a retrieval threshold before automatic handling



The agent should not automatically respond when there is no sufficiently similar historical case. A minimum retrieval score provides a simple safeguard against unsupported responses.



\## 14. Added explicit escalation rules



Messages containing escalation or high-risk complaint language are routed to humans. This intentionally favours safety over maximum automation coverage.



\## 15. Excluded support\_escalation from response-generation evaluation



Support-escalation cases are expected to be routed to humans rather than receive an automated customer reply. Response quality is therefore evaluated on intents that the current policy considers eligible for automatic handling.

