# Subagent Model Routing Design

**Date:** 2026-02-03
**Status:** Validated Design
**Purpose:** Cost-optimized model selection for Claude Code subagent spawning

---

## Problem Statement

When Claude Code (running on Opus) spawns subagents via the Task tool, it defaults to the parent model regardless of task complexity. This creates unnecessary costs:

- An Explore agent running `grep` on Opus costs 10-20x what Haiku would charge for identical results
- Simple file searches, bash commands, and basic code reviews don't need Opus-level reasoning
- No feedback loop exists to learn from routing decisions

**Goal:** Maximize output quality while minimizing cost through intelligent model routing.

---

## Solution Overview

A four-layer complexity routing system built into Claude Code's behavior:

```
┌─────────────────────────────────────────────────────────┐
│                    TASK DISPATCHED                       │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 1: Hard Rules Engine                             │
│  ───────────────────────                                │
│  Instant, zero-cost pattern matching                    │
│  Maps task types + keywords → model tier                │
│  ~70% of tasks get routed here                          │
└────────────────────────┬────────────────────────────────┘
                         │ No match
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 2: Haiku Classifier                              │
│  ─────────────────────────                              │
│  Fast, cheap LLM analysis of ambiguous tasks            │
│  Returns: model + confidence + reasoning                │
│  Confidence < 0.6 → bump up one tier                    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 3: Adaptive Retry                                │
│  ────────────────────────                               │
│  Monitors execution, escalates on failure               │
│  Explicit failure → auto-retry higher tier              │
│  Quality signals → prompt user before retry             │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: Adaptive Learning                             │
│  ───────────────────────                                │
│  Tracks outcomes, refines rules over time               │
│  Memory files + skill updates                           │
└─────────────────────────────────────────────────────────┘
```

**Enforcement:** Hybrid approach
- CLAUDE.md contains mandatory routing rules
- `complexity-router` skill provides decision tree for edge cases
- Pre-task verification hook warns on missing model selection

---

## Layer 1: Hard Rules Engine

**Purpose:** Route obvious cases instantly with zero overhead.

### Rules Matrix

| Signal Type | Pattern | → Model |
|-------------|---------|---------|
| **Agent Type** | `Explore` | Haiku |
| | `Bash` | Haiku |
| | `Plan` | Sonnet |
| | `code-reviewer`, `silent-failure-hunter` | Sonnet |
| | `general-purpose` | → Layer 2 |
| **Keywords (low)** | find, list, search, check, count, show | Haiku |
| **Keywords (high)** | design, architect, security, refactor entire, migrate | Opus |
| **Scope** | Single file mentioned | Haiku |
| | 2-5 files | Sonnet |
| | >5 files or "codebase", "all", "entire" | Sonnet+ |
| **Prompt length** | < 30 words + no ambiguity | Haiku |
| | > 200 words | Sonnet+ |

### Conflict Resolution

When multiple signals match, **highest tier wins**.

Example:
```
"Find all usages of AuthService"
  → "find" suggests Haiku
  → "all usages" suggests broader scope
  → Result: Sonnet (higher wins)
```

### Escape Hatch

If prompt contains explicit model request ("use opus for this"), honor it.

### CLAUDE.md Implementation

```markdown
## Subagent Model Selection (MANDATORY)

Before EVERY Task tool call, select model tier:
1. Check agent type against rules matrix
2. Scan prompt for complexity keywords
3. Assess scope (file count, codebase-wide)
4. Use highest matching tier
5. If no clear match → invoke complexity-router skill
```

---

## Layer 2: Haiku Classifier

**Purpose:** Handle ambiguous cases that don't match hard rules with intelligent, cheap analysis.

### When Invoked

- No hard rule matched
- Conflicting signals (e.g., "find" keyword but >10 files)
- general-purpose agent type (inherently ambiguous)

### Classifier Prompt

```
Analyze this task and recommend a model tier.

Task type: {agent_type}
Prompt: {prompt}

Consider:
- Complexity: Does this need deep reasoning or simple execution?
- Scope: How many files/components involved?
- Risk: What happens if we get it wrong?
- Ambiguity: Is the ask clear or does it require interpretation?

Return JSON:
{
  "model": "haiku|sonnet|opus",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
```

### Confidence Thresholds

```python
if confidence >= 0.6:
    use_model = classifier_recommendation
elif confidence >= 0.5:
    use_model = bump_tier(classifier_recommendation, +1)
elif confidence >= 0.3:
    use_model = "sonnet"  # safe middle ground
else:  # < 0.3
    use_model = "opus"  # genuinely confused, use best
```

### Cost Analysis

- Haiku call: ~500 tokens @ $0.0003 = $0.00015 per classification
- Savings from correct downgrade: Opus→Haiku = ~$0.02-0.10 per task
- **Break-even:** After 1-2 correct downgrades, classifier pays for itself

### Failure Handling

If Haiku classifier itself fails/times out → default to Sonnet (safe middle).

---

## Layer 3: Adaptive Retry

**Purpose:** Recover from routing mistakes by escalating to more capable models when tasks fail.

### Failure Detection

| Type | Signal | Action |
|------|--------|--------|
| **Explicit failure** | Error/exception, timeout, "I cannot" in response | Auto-retry +1 tier |
| **Quality signals** | Response < 50 words for >100 word prompt | Prompt user |
| | Hedging: "might", "possibly", "not sure" | Prompt user |
| | Incomplete code (missing functions, TODOs) | Prompt user |
| **User feedback** | User says "that's wrong" or "try again" | Retry with user-specified tier |

### Retry Policy

```
Attempt 1: Classifier/rule recommendation
Attempt 2: +1 tier (haiku→sonnet, sonnet→opus)
Attempt 3: Opus (final attempt)
Attempt 4: Fail with clear message to user
```

### Quality Check Prompt

When soft quality signals are detected:

```
The subagent completed but showed quality concerns:
- {detected_issues}

Options:
1. Accept this output (it might be fine)
2. Retry with Sonnet (better reasoning)
3. Retry with Opus (maximum capability)

Estimated additional cost: Sonnet +$0.02, Opus +$0.08
```

### Retry Logging

Every retry gets logged to adaptive learning memory:

```markdown
### Retry Event
- Task: "Analyze auth flow"
- Initial: haiku (rule: "analyze" keyword)
- Failure: Incomplete - missed OAuth integration
- Retry: sonnet → Success
- **Learning**: "analyze" + "flow"/"integration" → sonnet minimum
```

---

## Layer 4: Adaptive Learning

**Purpose:** Learn from routing outcomes to refine rules over time.

### What Gets Tracked

| Event | Data Captured |
|-------|---------------|
| **Routing decision** | task_type, prompt_hash, signals_matched, model_selected, confidence |
| **Execution outcome** | success/failure, retry_count, final_model_used, duration |
| **Quality signals** | output_length, hedging_detected, user_feedback |
| **Cost** | tokens_in, tokens_out, estimated_cost |

### Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│  RAW DATA (Memory files)                                │
│  ────────────────────────                               │
│  routing-outcomes.md - append-only log of decisions     │
│  Updated: after every subagent completion               │
│  Format: structured markdown, parseable                 │
└────────────────────────┬────────────────────────────────┘
                         │ Periodic analysis
                         ▼
┌─────────────────────────────────────────────────────────┐
│  LEARNED PATTERNS (Skill updates)                       │
│  ────────────────────────────────                       │
│  complexity-router skill gets updated when:             │
│  • A rule shows >20% failure rate over 50+ samples      │
│  • New keyword patterns emerge from successful routes   │
│  • Confidence thresholds prove too aggressive/lax       │
└─────────────────────────────────────────────────────────┘
```

### Learning Triggers

1. **Session end** - summarize outcomes, append to memory
2. **Weekly review** - (manual or prompted) analyze patterns, propose skill updates
3. **Threshold breach** - if retry rate spikes, flag for immediate review

### Example Memory Entry

```markdown
## 2026-02-03 Session

| Task | Initial | Final | Retries | Outcome |
|------|---------|-------|---------|---------|
| Explore: find auth files | haiku | haiku | 0 | ✓ |
| Review: security audit | sonnet | opus | 1 | ✓ |
| Plan: API redesign | sonnet | sonnet | 0 | ✓ |

Learnings:
- "security" keyword underweighted - escalate to opus default
```

---

## Enforcement Mechanism

### CLAUDE.md Entry

```markdown
## MANDATORY: Subagent Model Selection

You MUST select the appropriate model tier before EVERY Task tool call.

### Decision Flow
1. Check agent type against hard rules (Explore→Haiku, Plan→Sonnet)
2. Scan prompt for complexity keywords
3. Check scope indicators (file count, "entire codebase")
4. If clear match → set model parameter
5. If ambiguous → invoke `complexity-router` skill
6. NEVER default to parent model without justification

### Hard Rules Quick Reference
- Explore, Bash → haiku
- Plan, code-reviewer → sonnet
- find/list/search + single file → haiku
- design/architect/security → opus
- Ambiguous → skill invocation required
```

### Skill Structure

File: `~/.claude/plugins/complexity-router.md`

```markdown
---
name: complexity-router
description: Determines optimal model tier for subagent tasks
---

## When to Use
Invoke BEFORE spawning any subagent when hard rules don't apply.

## Process
1. If task matches hard rule → return recommendation immediately
2. Otherwise → spawn Haiku classifier
3. Apply confidence thresholds
4. Log decision to memory
5. Return model recommendation

## Output
"Use {model} for this task because {reasoning}"
```

### Verification Hook

File: `.claude/hooks/pre-task.sh`

```bash
#!/bin/bash
# Warn if model parameter is missing from Task call

if ! grep -q '<parameter name="model">' <<< "$TOOL_CALL"; then
  echo "⚠️  Warning: No model specified for Task call. Did you apply routing rules?"
fi
```

---

## Metrics & Monitoring

### Cost Metrics

```markdown
### Daily/Weekly Reports
- Total subagent calls: 247
- Model distribution: Haiku 65%, Sonnet 28%, Opus 7%
- Total cost: $2.34 (vs $18.50 without routing = 87% savings)
- Cost per task type:
  - Explore: $0.002 avg
  - Plan: $0.015 avg
  - code-reviewer: $0.025 avg
```

### Quality Metrics

```markdown
### Success Rates by Initial Routing
- Haiku → Success: 89% (142/160)
- Sonnet → Success: 94% (65/69)
- Opus → Success: 100% (18/18)

### Retry Statistics
- Tasks requiring retry: 23 (9.3%)
- Haiku→Sonnet: 18
- Sonnet→Opus: 5
- Cost of retries: $0.47 (20% of total)
```

### Learning Metrics

```markdown
### Pattern Detection
- New rules identified: 3
  - "migration" keyword → sonnet (was haiku, 60% fail rate)
  - "entire codebase" + Explore → sonnet (was haiku, 40% fail)
- Rules retired: 1
  - "check" → haiku proved too broad, now context-dependent
```

### Dashboard Location

`~/.claude/metrics/routing-report-{YYYY-MM-DD}.md`

Generated via: `orch analyze-routing` command (future enhancement)

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- Write CLAUDE.md routing rules
- Create `complexity-router` skill
- Add pre-task verification hook
- Test with manual model selection

### Phase 2: Classifier (Week 2)
- Implement Haiku classifier prompt
- Add confidence threshold logic
- Test on ambiguous tasks
- Validate cost savings

### Phase 3: Retry Logic (Week 3)
- Implement failure detection
- Add quality signal detection
- Create user prompts for soft failures
- Test escalation paths

### Phase 4: Learning (Week 4)
- Create memory file structure
- Implement outcome logging
- Build analysis scripts
- Test pattern detection

### Phase 5: Metrics (Ongoing)
- Dashboard generation
- Weekly reports
- Rule refinement based on data

---

## Expected Outcomes

**Cost Savings:**
- Target: 70-85% reduction in subagent costs
- Based on model distribution: 65% Haiku, 28% Sonnet, 7% Opus

**Quality:**
- First-attempt success rate: >85%
- Retry rate: <15%
- User intervention rate: <5%

**Learning:**
- New rules identified: 2-3 per month
- Rule accuracy improvement: +5-10% over 3 months

---

## Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| **Haiku classifier** | Better nuance than rules | +$0.00015 per call, +200ms latency |
| **Adaptive retry** | Recover from mistakes | +20% cost on failed attempts |
| **Learning system** | Self-improving | Memory overhead, periodic analysis |
| **Verification hook** | Prevents accidental defaults | Adds friction to workflow |

---

## Future Enhancements

1. **User preferences** - learn individual user's cost/quality trade-off preferences
2. **Task caching** - similar prompts → reuse previous model selection
3. **Multi-model comparison** - run same task on multiple models, use cheapest acceptable
4. **Orch integration** - extend to external AI CLIs (Gemini, Codex)
5. **Real-time dashboards** - live cost tracking during sessions

---

## Conclusion

This design balances cost efficiency with quality through a multi-layered approach:
- Hard rules handle the obvious (70% of cases)
- Haiku classifier handles ambiguity cheaply
- Adaptive retry provides a safety net
- Learning system continuously improves

The system is self-improving, transparent, and gives users visibility into routing decisions and costs.
