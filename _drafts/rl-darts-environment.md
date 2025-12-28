---
title: "Building an RL Environment for Darts"
date: 2025-12-15 10:00:00 +0000
categories: [Projects, Reinforcement Learning]
tags: [rl, environment, darts, gymnasium]
math: true
published: false
---

## Overview

Designing a Gymnasium-compatible RL environment for dart throwing—modeling physics, strategy, and skill acquisition.

---

## Motivation

- **Why darts?** Simple physics, complex strategy, interesting learning dynamics
- **RL challenges**: Sparse rewards, continuous action space, partial observability
- **Goal**: Train agents to develop human-like dart strategies

---

## Environment Design

### State Space

What does the agent observe?

- Current score (501 down)
- Dart position history
- Player statistics (accuracy, consistency)
- Game phase (early/mid/end game)

$$
s_t = (\text{score}_t, \text{history}_{t-k:t}, \text{stats}_t, \text{phase}_t)
$$

### Action Space

How does the agent throw?

- **Continuous**: $(x, y, \text{power})$ targeting coordinates
- **Discrete**: Predefined checkout paths
- **Hybrid**: Strategic target selection + execution noise

### Reward Function

$$
r(s, a, s') = \begin{cases}
+100 & \text{game won} \\
+10 \cdot \text{score\_diff} & \text{progress toward checkout} \\
-1 & \text{bust (invalid score)} \\
0 & \text{otherwise}
\end{cases}
$$

---

## Physics Simulation

### Throw Dynamics

Key factors to model:

1. **Release angle and velocity**
2. **Drag coefficient** (dart design matters)
3. **Board bounce** (scoring zones)
4. **Skill noise** (human-like variation)

```python
class DartPhysics:
    def __init__(self, skill_level=0.8):
        self.skill = skill_level
        self.drag = 0.01
        
    def simulate_throw(self, target, power):
        # Add skill-based noise
        noise = np.random.normal(0, 1 - self.skill, size=2)
        actual_target = target + noise
        
        # Simulate trajectory
        hit_position = self.calculate_trajectory(
            actual_target, power, self.drag
        )
        
        # Determine score
        score = self.board.score_at(hit_position)
        return score, hit_position
```

---

## Strategy Components

### Early Game (501 → ~170)

- Maximize average score per dart
- Triple 20 (T20) optimal
- Learn trade-offs between risk and consistency

### Mid Game (170 → ~100)

- Position for checkout
- Target specific doubles
- Manage "bust" risk

### End Game (<100)

- Checkout paths (e.g., 60 → T20, D20)
- Conditional strategies based on darts remaining

---

## Training Approach

### Curriculum Learning

1. **Stage 1**: Learn basic throwing mechanics (hit the board)
2. **Stage 2**: Score optimization (maximize points)
3. **Stage 3**: Strategic play (checkout sequences)

### Algorithms to Try

- **PPO**: Standard continuous control baseline
- **SAC**: Explore stochastic policies
- **Hierarchical RL**: High-level strategy + low-level execution

---

## Implementation Notes

Key challenges encountered:

- [ ] Defining realistic physics parameters
- [ ] Balancing exploration vs. exploitation in sparse reward setting
- [ ] Modeling human-like skill progression

---

## Next Steps

1. Implement basic Gymnasium environment
2. Validate physics with real-world dart data
3. Train baseline PPO agent
4. Add curriculum learning stages

---

_Draft in progress. Code and results coming soon._
