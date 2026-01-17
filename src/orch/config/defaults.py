"""Default configuration values."""

# Default agents to use in competition mode
DEFAULT_COMPETITION_AGENTS = ["gemini", "codex"]

# Task type routing defaults
DEFAULT_ROUTING_RULES = {
    "code": ["codex", "gemini"],
    "explain": ["gemini", "codex"],
    "debug": ["codex", "gemini"],
    "general": ["gemini", "codex"],
}

# Keywords for task classification
DEFAULT_ROUTING_KEYWORDS = {
    "code": [
        "implement",
        "write code",
        "function",
        "class",
        "refactor",
        "create",
        "build",
        "add feature",
        "method",
        "module",
    ],
    "explain": [
        "explain",
        "what is",
        "how does",
        "why",
        "describe",
        "tell me about",
        "understand",
    ],
    "debug": [
        "error",
        "bug",
        "fix",
        "traceback",
        "exception",
        "failing",
        "broken",
        "not working",
        "issue",
    ],
}

# Evaluation criteria for competition mode
DEFAULT_EVALUATION_CRITERIA = [
    "correctness",
    "efficiency",
    "cleanliness",
    "architecture",
]

# Maximum iterations for delegation feedback loop
DEFAULT_MAX_ITERATIONS = 3

# tmux defaults
DEFAULT_TMUX_SESSION_NAME = "orch-session"
DEFAULT_TMUX_LAYOUT = "tiled"
