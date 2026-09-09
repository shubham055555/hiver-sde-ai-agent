\# Amazon Customer Support AI Agent



An AI support agent built from historical AmazonHelp conversations on Twitter. The system classifies incoming customer messages, retrieves similar historical support cases, drafts a grounded response, and decides whether the case should be automatically handled or escalated to a human.



\## Problem



The goal is to simulate an AI customer-support workflow using real historical customer-brand conversations.



For each incoming customer message, the system:



1\. Predicts the customer's support intent.

2\. Retrieves similar historical Amazon support cases.

3\. Uses those cases to ground a draft response.

4\. Decides between automatic handling and human escalation.

5\. Provides a reason for the decision.



The system is intentionally designed as a small, explainable prototype rather than a production support system.



\## What is built



\- AmazonHelp-specific conversation corpus

\- 11-intent customer support taxonomy

\- TF-IDF + Logistic Regression intent classifier

\- TF-IDF cosine-similarity historical case retrieval

\- Gemini-based grounded response generation

\- Rule-based human escalation layer

\- 200-example hand-labelled golden evaluation set

\- Retrieval evaluation

\- Decision-layer evaluation

\- Human response evaluation rubric

\- LLM-as-judge evaluation pipeline

\- Baseline comparison

\- Failure analysis

\- Decision log



\## What is not built



This prototype does not include:



\- live Twitter/X integration

\- live Amazon order lookup

\- access to customer accounts or order data

\- real refund or cancellation execution

\- production authentication

\- production monitoring

\- large-scale model fine-tuning

\- a production-grade safety or compliance system



The generated response is therefore a draft and not an action taken on a real customer account.



\## Dataset



The primary dataset is the Customer Support on Twitter dataset.



The full dataset contains approximately 2.8 million tweets and includes inbound customer messages, outbound brand responses, and tweet relationship fields.



AmazonHelp was selected because it had the largest number of brand-to-customer interactions among the candidate brands examined.



The AmazonHelp subset contains:



\- 82,556 reconstructed conversations

\- 374,042 messages

\- 51,260 conversations with at least three messages



Very long conversations containing more than 20 messages were removed from the retrieval corpus to reduce noise from long multi-issue threads.



\## Conversation reconstruction



Tweets were linked using:



\- `tweet\_id`

\- `response\_tweet\_id`

\- `in\_response\_to\_tweet\_id`



This allows customer messages and Amazon responses to be reconstructed into conversation threads instead of treating every tweet independently.



For response grounding, the retrieval corpus focuses on customer messages that were directly followed by an AmazonHelp response.



This produced 142,201 historical customer-response pairs.



\## Intent taxonomy



The taxonomy contains 11 customer intents:



| Intent | Description |

|---|---|

| delivery\_issue | Late, missing, damaged, wrong, or incomplete delivery |

| order\_management | Tracking, cancellation, modification, or preorder issues |

| returns\_refunds | Returns, refunds, replacements, or money-back requests |

| payment\_billing | Charges, payment methods, taxes, or billing |

| prime\_subscription | Prime membership, subscription, or benefits |

| account\_access | Login, password, or account access |

| product\_technical | Product or device setup and functionality |

| product\_information | Product availability, features, compatibility, or general information |

| seller\_review | Seller-specific complaints or reviews |

| support\_escalation | Unresolved support issues or explicit escalation requests |

| other\_non\_support | Casual, unclear, or non-actionable messages |



Labels represent customer intent rather than product category.



\## Architecture



```text

Customer message

&#x20;      |

&#x20;      v

Intent classifier

&#x20;      |

&#x20;      v

Historical case retrieval

&#x20;      |

&#x20;      +------------------+

&#x20;      |                  |

&#x20;      v                  v

Retrieval score      Predicted intent

&#x20;      |                  |

&#x20;      +--------+---------+

&#x20;               |

&#x20;               v

&#x20;       Decision layer

&#x20;         /        \\

&#x20;        /          \\

&#x20;       v            v

Auto-handle      Human escalation

&#x20;    |

&#x20;    v

Gemini response generation

&#x20;    |

&#x20;    v

Grounded draft reply

