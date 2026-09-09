def make_decision(intent, message, retrieval_score):
    message = message.lower()

    escalation_words = [
        "manager",
        "supervisor",
        "complaint",
        "lawyer",
        "legal",
        "chargeback",
        "fraud",
        "urgent",
        "escalate"
    ]

    if any(word in message for word in escalation_words):
        return {
            "decision": "human_escalation",
            "reason": "Customer explicitly requests escalation or indicates a high-risk complaint."
        }

    if intent == "support_escalation":
        return {
            "decision": "human_escalation",
            "reason": "The issue appears unresolved or requires human support intervention."
        }

    if intent == "other_non_support":
        return {
            "decision": "human_escalation",
            "reason": "The message does not contain enough actionable support information for safe automation."
        }

    if retrieval_score < 0.25:
        return {
            "decision": "human_escalation",
            "reason": "No sufficiently similar historical case was retrieved."
        }

    return {
        "decision": "auto_handle",
        "reason": "The intent is supported and a sufficiently similar historical case is available."
    }


if __name__ == "__main__":
    tests = [
        ("delivery_issue", "My package is late", 0.65),
        ("returns_refunds", "I want a refund", 0.52),
        ("support_escalation", "I want to speak to a manager", 0.60),
        ("product_information", "Is this product compatible?", 0.15)
    ]

    for intent, message, score in tests:
        result = make_decision(intent, message, score)

        print("\nMessage:", message)
        print("Decision:", result["decision"])
        print("Reason:", result["reason"])