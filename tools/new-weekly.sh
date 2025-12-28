#!/bin/bash
# tools/new-weekly.sh - Create a new weekly notes post

# Get current date
DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H:%M:%S")
WEEK_START=$(date -v-Mon +"%b %d")
WEEK_END=$(date -v+Sun +"%b %d, %Y")

# Create filename
FILENAME="_posts/${DATE}-weekly-notes.md"

# Check if file exists
if [ -f "$FILENAME" ]; then
    echo "❌ Weekly notes for this week already exist: $FILENAME"
    exit 1
fi

# Copy template and update date
cat > "$FILENAME" << EOF
---
title: "Weekly Notes: ${WEEK_START} – ${WEEK_END}"
date: ${DATE} ${TIME} +0000
categories: [Weekly Notes]
tags: [weekly, training, observations]
math: true
---

## Week of ${WEEK_START} – ${WEEK_END}

_Brief summary of the week's focus and key themes._

---

## Training Observations

### [Specific Topic or Experiment]

**Context**: What were you working on?

**Observations**:
- Key finding 1
- Key finding 2
- Key finding 3

**Metrics**:
- Relevant numbers, losses, accuracies, etc.

**Code Snippet** (if relevant):

\`\`\`python
# Example implementation or key algorithm
\`\`\`

---

## Paper Notes

### [Paper Title](link)

**Main Contribution**: One-line summary

**Key Ideas**:
- Insight 1
- Insight 2
- Practical implication

**Implementation Notes**:
- How to apply this
- Pitfalls to avoid

---

## Random Thoughts

- Observation about training dynamics
- Interesting bug or behavior
- Tool or technique worth noting

---

## Next Week

- [ ] Task 1
- [ ] Experiment 2
- [ ] Paper to read

---

_Weekly notes are meant to be rough, informal, and capture-in-the-moment. No polish required._
EOF

echo "✅ Created new weekly notes: $FILENAME"
echo "📝 Open it with: code $FILENAME"
