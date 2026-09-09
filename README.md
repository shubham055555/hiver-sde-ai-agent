# AI Customer Support Agent

An AI customer-support agent built for the Hiver SDE Intern take-home assignment.

The system takes an incoming Amazon customer message and performs three tasks:

1. Predicts the customer's support intent.
2. Retrieves similar historical Amazon support cases and uses them to ground a draft reply.
3. Decides whether the case can be auto-handled or should be escalated to a human.

The project focuses on a single brand, AmazonHelp, from the Customer Support on Twitter dataset.

## Problem framing

Customer-support conversations contain repeated issue patterns and examples of how support agents previously responded. The goal is to use those historical interactions to build a lightweight support agent that can classify incoming requests, suggest grounded responses, and avoid automatically handling cases that require human intervention.

The system is designed as a prototype rather than a production customer-support system.

### What is built

* Customer intent classification
* Historical-case retrieval
* LLM-based grounded reply generation
* Auto-handle vs human-escalation decision layer
* 200-example golden evaluation set
* Classification baselines
* Retrieval evaluation
* Decision-layer evaluation
* Response-quality rubric
* LLM-as-judge evaluation pipeline
* Human evaluation pipeline
* Failure-mode analysis
* Decision log

### What is not built

* Production deployment
* Real Amazon APIs
* Real order lookup
* Authentication or customer identity verification
* Real refund or cancellation execution
* Fine-tuning of a large language model
* Production-grade multilingual support
* Fully calibrated production safety thresholds
* Human-agent feedback loop

## Data source

The primary dataset is the Customer Support on Twitter dataset from Kaggle:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The dataset contains customer-support conversations between customers and brands on Twitter.

This project uses the AmazonHelp subset. Conversation threads are reconstructed using the tweet response relationships in the dataset.

The full raw dataset is intentionally not included in the repository because of its size.

## Dataset exploration

The original dataset contains approximately 2.8 million tweets.

For the selected AmazonHelp brand, the project identified:

* 169,840 outbound AmazonHelp tweets
* 100,503 direct customer replies to AmazonHelp tweets

Conversation threads were reconstructed from:

* `tweet_id`
* `response_tweet_id`
* `in_response_to_tweet_id`
* `inbound`
* `created_at`

The reconstructed AmazonHelp dataset contains:

* 82,556 conversation roots
* 374,042 messages
* 51,260 conversations with at least three messages

The median conversation length is three messages.

Very long conversations can dominate retrieval and create unusually repetitive examples, so conversations containing more than 20 messages were removed from the modelling corpus.

After this filtering:

* 81,726 conversations remained
* 344,748 messages remained

## Intent taxonomy

The intent taxonomy was created from the observed AmazonHelp customer-support messages.

The labels describe the customer's intent rather than simply the product category.

| Intent                | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| `delivery_issue`      | Late, missing, damaged, wrong or incomplete delivery                  |
| `order_management`    | Tracking, cancellation, modification or order-management requests     |
| `returns_refunds`     | Returns, refunds, replacements and money-back requests                |
| `payment_billing`     | Charges, payment methods, taxes and billing problems                  |
| `prime_subscription`  | Prime membership, subscription and Prime benefits                     |
| `account_access`      | Login, password and account-access problems                           |
| `product_technical`   | Product or device setup and functionality problems                    |
| `product_information` | Product availability, compatibility, features and general information |
| `seller_review`       | Seller-related or review-related messages                             |
| `support_escalation`  | Unresolved cases or explicit requests for escalation                  |
| `other_non_support`   | Casual, unclear or non-support messages                               |

## Golden evaluation set

A manually labelled golden set of 200 customer messages was created.

The examples were sampled from the AmazonHelp customer-message corpus and labelled according to the intent taxonomy above.

The set contains all 11 intents, although the distribution is intentionally not uniform because it reflects the sampled customer-message distribution.

Current distribution:

| Intent                | Examples |
| --------------------- | -------: |
| `delivery_issue`      |       68 |
| `support_escalation`  |       30 |
| `other_non_support`   |       26 |
| `returns_refunds`     |       19 |
| `product_information` |       12 |
| `prime_subscription`  |       12 |
| `product_technical`   |       10 |
| `payment_billing`     |        9 |
| `order_management`    |        6 |
| `account_access`      |        6 |
| `seller_review`       |        2 |

The small number of examples in rare classes is an important limitation of the evaluation.

## System architecture

The current pipeline is:

```text
Customer message
       |
       v
Intent classifier
       |
       +----------------------+
       |                      |
       v                      v
Historical retrieval      Decision layer
       |                      |
       v                      v
Similar Amazon cases     Auto-handle /
       |                 Human escalation
       v
Gemini 3.6 Flash
       |
       v
Grounded draft reply
```

The response generator is only called when the decision layer selects `auto_handle`.

## Intent classification

The first baseline uses TF-IDF features with word unigrams and bigrams.

Configuration:

* `TfidfVectorizer`
* n-grams: `(1, 2)`
* minimum document frequency: `2`
* maximum features: `10,000`
* Logistic Regression
* `class_weight="balanced"`

A stratified five-fold evaluation was used for the reported classification result.

### Classification results

| Model                        | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| Majority-class baseline      |    34.0% |    4.61% |
| TF-IDF + Logistic Regression |    36.0% |   19.15% |

The majority baseline always predicts the most common intent, `delivery_issue`.

The TF-IDF model improves over the majority baseline, but the low macro F1 shows that performance is highly uneven across intents.

The classifier performs best on common intents such as delivery issues and returns/refunds. Rare intents frequently receive poor F1 scores because the golden set contains very few examples for them.

## Retrieval

For response grounding, the system retrieves historical customer messages from AmazonHelp conversations.

A retrieval pair consists of:

```text
customer message -> AmazonHelp response
```

The current retrieval implementation uses:

* TF-IDF word unigrams and bigrams
* maximum 50,000 features
* cosine similarity

The top three historical cases are passed to the response-generation model.

For example, for a customer message about a late package, the system retrieves historical cases where AmazonHelp asked for delivery-date or tracking information or provided delivery-related guidance.

The purpose of retrieval is not to copy previous responses. It is to provide examples of how similar cases were historically handled.

### Retrieval evaluation

The 200-example golden set was used to measure whether retrieved cases have the same intent as the current example.

The current result is:

* Top-1 intent agreement: 84.5%
* Top-3 intent agreement: 92.0%

This is a retrieval **intent-agreement proxy**, not ground-truth retrieval accuracy.

The historical retrieval examples do not have manually verified intent labels. Their intents are inferred using the classifier, so this metric should not be interpreted as 84.5% true retrieval accuracy.

## Response generation

Gemini 3.6 Flash is used to generate the final draft response.

The model receives:

* Customer message
* Predicted intent
* Top three historical customer/support examples

The prompt instructs the model to:

* use historical examples as grounding
* avoid inventing order information
* avoid inventing refunds or delivery dates
* avoid inventing policies or actions
* ask only for information needed to investigate the issue
* not mention the historical examples
* not mention that the response was generated by AI

A representative response for a late-package message is structured around asking for the expected delivery date and latest tracking update, because those are the types of information requested in similar historical AmazonHelp cases.

## Response evaluation

Response generation is evaluated separately from intent classification.

A 20-example response evaluation set was constructed using two examples from each of ten non-escalation intents.

`support_escalation` is excluded from response generation evaluation because the intended behavior for those cases is human escalation rather than automated reply generation.

The response rubric contains five dimensions scored from 1 to 5:

1. Groundedness
2. Helpfulness
3. Intent correctness
4. Absence of unsupported claims
5. Overall quality

A response is considered acceptable when all critical dimensions meet the required threshold.

Two evaluation paths are implemented:

* Human evaluation
* LLM-as-judge evaluation using Gemini 3.6 Flash

The repository contains the evaluation scripts and rubric, but final judge-human agreement should only be reported after both evaluations have been run on the generated response set.

## Auto-handle vs human escalation

The decision layer combines:

* predicted intent
* customer message content
* retrieval similarity

The current rules escalate when:

* the customer explicitly requests a manager, supervisor or escalation
* the message contains high-risk complaint language such as fraud, legal or chargeback
* the predicted intent is `support_escalation`
* the best retrieval score is below the current threshold

`other_non_support` messages are also escalated because there may not be enough actionable support information for safe automation.

Otherwise, the case is marked `auto_handle`.

### Decision results

On the current 200-example evaluation set:

* 132 cases were marked `auto_handle`
* 68 cases were marked `human_escalation`

This corresponds to:

* Auto-handle: 66%
* Human escalation: 34%

These results are behavioral outputs of the current rule-based decision layer, not production safety guarantees.

## Failure modes

The main observed failure modes are:

### 1. Multilingual and very short messages

Short Spanish and Portuguese messages can be classified incorrectly because the classifier is trained mainly on English-language support text.

Example:

```text
@115821 necesito de su ayuda..
```

The expected intent is `support_escalation`, but the classifier predicts `product_information`.

A multilingual model or multilingual embeddings would likely improve this.

### 2. Delivery issues mixed with escalation language

A customer can describe a normal delivery problem while also using words such as "urgent" or "complaint".

The current rule-based decision layer may escalate these cases even when the underlying issue could be handled automatically.

### 3. Low retrieval confidence

A low retrieval score can cause human escalation even when the predicted intent is otherwise reasonable.

This makes the system conservative but can also increase unnecessary escalations.

### 4. Mixed-intent messages

Some messages contain multiple issues, such as a delivery problem combined with payment or legal language.

The single-label intent taxonomy forces the classifier to choose one label, which can lose important information.

### 5. Casual or informational messages

Messages such as announcements, thanks or unclear requests can be confused with support intents.

The `other_non_support` category helps reduce unsafe automation, but the current classifier does not always separate these messages cleanly.

## What is misleading about my headline number?

A tempting headline number is the 94% accuracy obtained during an early experiment where the classifier was trained and evaluated on the same 200 golden examples.

That number is misleading because the evaluation examples were also used for training.

When the same 200 examples are evaluated using a leakage-free five-fold procedure, performance drops to:

* 36.0% accuracy
* 19.15% macro F1

The golden set is also highly imbalanced, with only two examples for `seller_review`.

The retrieval number also needs careful interpretation. The reported 84.5% top-1 result is an intent-agreement proxy based partly on classifier predictions, not a manually verified retrieval accuracy.

These limitations are more representative of the current system than the early 94% number.

## What I would build next in one week

### 1. Strengthen the evaluation set

Increase the golden set from 200 examples to at least 500-1,000 examples, with minimum coverage requirements for every intent.

### 2. Improve intent classification

Compare the current TF-IDF classifier against a stronger pretrained text classifier or embedding-based approach.

The goal would be to improve minority-intent recall rather than optimizing only overall accuracy.

### 3. Improve retrieval

Compare:

* TF-IDF retrieval
* dense embedding retrieval
* hybrid retrieval
* retrieval followed by a cross-encoder reranker

The reranker could select the most relevant historical support example after a broader candidate retrieval stage.

### 4. Calibrate the decision layer

Instead of using fixed keyword rules and a single retrieval threshold, evaluate escalation decisions against manually labelled human-vs-automation decisions.

This would allow the escalation threshold to be optimized for an explicit safety/coverage trade-off.

### 5. Support mixed and multilingual messages

Move from single-label classification toward multi-label intent detection where appropriate and use multilingual representations for non-English customer messages.

## Reproducibility

The repository intentionally excludes the full raw and processed datasets because of their size.

To reproduce the project:

### 1. Clone the repository

```bash
git clone https://github.com/shubham055555/hiver-sde-ai-agent.git
cd hiver-sde-ai-agent
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install pandas scikit-learn python-dotenv google-genai
```

### 4. Download the dataset

Download the Customer Support on Twitter dataset from Kaggle and place the extracted files under:

```text
data/raw/
```

The main dataset should be available at:

```text
data/raw/twcs/twcs.csv
```

### 5. Reconstruct AmazonHelp conversations

```powershell
python src/data/reconstruct_threads.py
python src/data/build_thread_dataset.py
python src/data/quality_check.py
python src/data/prepare_corpus.py
python src/data/clean_text.py
```

### 6. Build retrieval pairs

```powershell
python src/data/build_retrieval_pairs.py
```

### 7. Evaluate classification

```powershell
python src/data/evaluate_baselines.py
python src/model/evaluate_classifier.py
```

### 8. Evaluate retrieval

```powershell
python src/data/evaluate_retrieval.py
```

### 9. Run the agent

Set the Gemini API key in `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

Then run:

```powershell
python src/agent/agent.py
```

Enter a customer-support message when prompted.

The agent returns:

* predicted intent
* auto-handle or human escalation
* escalation reason
* retrieval score
* generated draft reply when auto-handled

## Project structure

```text
hiver-sde-ai-agent/
|
|-- data/
|   |-- golden/
|   |   |-- golden_set.csv
|   |   `-- response_eval_set.csv
|   |
|   |-- raw/
|   |   `-- twcs/
|   |       `-- twcs.csv
|   |
|   `-- processed/
|
|-- src/
|   |-- agent/
|   |   |-- agent.py
|   |   |-- decision.py
|   |   |-- draft_reply.py
|   |   `-- test_gemini.py
|   |
|   |-- data/
|   |   |-- analyze_brands.py
|   |   |-- analyze_decisions.py
|   |   |-- analyze_threads.py
|   |   |-- build_conversations.py
|   |   |-- build_response_eval.py
|   |   |-- build_retrieval_pairs.py
|   |   |-- build_thread_dataset.py
|   |   |-- clean_text.py
|   |   |-- compare_brands.py
|   |   |-- create_golden_set.py
|   |   |-- decision_log.md
|   |   |-- evaluate_baselines.py
|   |   |-- evaluate_human_responses.py
|   |   |-- evaluate_retrieval.py
|   |   |-- explore_intents.py
|   |   |-- find_decision_failures.py
|   |   |-- generate_response_eval.py
|   |   |-- inspect_conversations.py
|   |   |-- inspect_intents.py
|   |   |-- llm_judge.py
|   |   |-- prepare_corpus.py
|   |   |-- quality_check.py
|   |   |-- reconstruct_threads.py
|   |   |-- report_findings.md
|   |   |-- response_rubric.md
|   |   |-- retrieve_similar.py
|   |   `-- sample_customer_messages.py
|   |
|   |-- eval_agent.py
|   |-- eval_agent_behavior.py
|   |-- eval_agent_decisions.py
|   |-- label_golden_set.py
|   |
|   `-- model/
|       |-- evaluate_classifier.py
|       `-- train_classifier.py
|
|-- .gitignore
`-- README.md
```

## Decision log

The non-obvious implementation decisions are documented in:

```text
src/data/decision_log.md
```

The main decisions include:

* Selecting AmazonHelp as the primary brand
* Reconstructing conversations rather than treating tweets independently
* Removing extremely long conversations from the modelling corpus
* Defining intents around customer intent
* Creating a manually labelled golden set
* Using a majority-class baseline
* Using TF-IDF plus Logistic Regression as the simple classifier baseline
* Using five-fold evaluation instead of same-set evaluation
* Retrieving historical customer-response pairs
* Using retrieval intent agreement as a proxy metric
* Introducing a retrieval confidence threshold
* Adding explicit escalation rules
* Excluding `support_escalation` from response-generation evaluation

## Limitations

The current prototype has several limitations:

* The golden set is relatively small.
* Several intents have very few examples.
* The classifier is a simple TF-IDF model.
* Retrieval uses lexical similarity rather than semantic embeddings.
* The decision layer is rule-based.
* Multilingual messages are not handled robustly.
* Response-generation evaluation still requires both human and LLM judging before judge-human agreement can be reported.
* The historical dataset reflects Twitter support behavior and may not represent current Amazon support policies.
* The system does not execute real customer-support actions.

The goal of this project is therefore to demonstrate a reproducible support-agent architecture and honest evaluation process rather than claim production readiness.
