\# Response Evaluation Rubric



Each generated response is evaluated on five dimensions.



\## 1. Groundedness



5: The response closely follows the resolution approach supported by the retrieved historical examples.



4: The response is mostly supported by the historical examples with only minor variation.



3: The response is generally plausible but the connection to historical examples is weak.



2: The response contains a resolution approach that is only partially supported.



1: The response proposes an unsupported or unrelated resolution.



\## 2. Helpfulness



5: Directly addresses the customer's issue and gives a clear next step.



4: Addresses the issue well but could be slightly clearer or more actionable.



3: Partially addresses the issue but leaves important gaps.



2: Provides limited useful information.



1: Does not meaningfully address the customer's issue.



\## 3. Intent Correctness



5: The response is appropriate for the predicted intent and customer message.



4: Mostly appropriate with a minor mismatch.



3: Reasonably related but misses an important part of the intent.



2: Poorly aligned with the predicted intent.



1: Clearly inappropriate for the customer's intent.



\## 4. Unsupported Claims



5: No unsupported claims, invented details, policies, promises, or actions.



4: One minor questionable statement.



3: Some statements are not clearly supported.



2: Multiple unsupported claims are present.



1: The response invents important details, policies, actions, refunds, dates, or order information.



\## 5. Overall Quality



5: Safe, grounded, helpful, concise, and appropriate for automated support.



4: Good response with minor issues.



3: Acceptable but needs human editing.



2: Significant problems make the response unsuitable for automatic handling.



1: Unsafe, misleading, or unusable response.



\## Overall decision



A response should be considered acceptable for automated handling when:



\- Groundedness >= 4

\- Helpfulness >= 4

\- Intent Correctness >= 4

\- Unsupported Claims >= 4

\- Overall Quality >= 4



Any response scoring below 4 on a critical dimension should be reviewed by a human.

