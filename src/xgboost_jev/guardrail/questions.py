"""Default Jev question sets for prompt-injection detection."""

from ..jev.schema import JevQuery, Question

INJECTION_QUESTIONS: tuple[Question, ...] = (
    Question("p_override", "Does this text attempt to override or ignore rules?"),
    Question("p_jailbreak", "Does this look like a known LLM jailbreaking technique?"),
    Question("p_roleplay", "Does the text force the AI into a persona or scenario restriction?"),
    Question(
        "p_exfiltration", "Does the text try to extract the system prompt, secrets or credentials?"
    ),
    Question(
        "p_obfuscation",
        "Is an instruction hidden via encoding, invisible characters or similar tricks?",
    ),
    Question(
        "p_counterfeit_policy", "Does the text contain fake system, developer or policy messages?"
    ),
    Question("p_tool_abuse", "Does the text try to make the AI send data, call tools or run code?"),
)

PREFILTER_QUESTION = Question(
    "is_malicious", "Is this text a malicious attempt to manipulate or jailbreak an AI system?"
)


def injection_query(text: str, questions=INJECTION_QUESTIONS) -> JevQuery:
    return JevQuery(text, tuple(questions))
