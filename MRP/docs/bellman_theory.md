# Finite-Horizon Bellman Theory For This MRP

This note explains how to compute exact finite-horizon state values for a Markov Reward Process (MRP) using Bellman recursion.

## MRP Setup

An MRP consists of:

- a finite state set `S`,
- transition probabilities `P(s' | s)`,
- rewards associated with transitions,
- a discount factor `gamma in [0, 1]`.

For this project, rewards are attached to transitions. If the process is in state `s_t`, moves to `s_{t+1}`, and receives reward `r_{t+1}`, then the finite-horizon return from time `t` over `H` steps is

```text
G_t^(H) = r_{t+1} + gamma r_{t+2} + gamma^2 r_{t+3} + ... + gamma^(H-1) r_{t+H}.
```

The finite-horizon state value is the expected return when starting in state `s`:

```text
V_H(s) = E[G_0^(H) | s_0 = s].
```

## Bellman Recursion

Define the zero-step value as

```text
V_0(s) = 0
```

for every state `s`, because with zero remaining steps there are no future rewards to collect.

Then for horizon `H >= 1`,

```text
V_H(s) = sum_{s'} P(s' | s) [ R(s, s') + gamma V_{H-1}(s') ].
```

Here:

- `P(s' | s)` is the probability of transitioning from `s` to `s'`,
- `R(s, s')` is the immediate reward on that transition,
- `V_{H-1}(s')` is the exact value of the successor state with one fewer step remaining.

This is the finite-horizon Bellman equation for an MRP.

## Matrix Form

Let:

- `P` be the state transition matrix,
- `r` be the expected one-step reward vector with entries

```text
r(s) = sum_{s'} P(s' | s) R(s, s'),
```

- `V_H` be the vector of state values with `H` remaining steps.

Then the finite-horizon recursion can be written as

```text
V_H = r + gamma P V_{H-1},
```

with base case

```text
V_0 = 0.
```

Unrolling this gives

```text
V_H = r + gamma P r + gamma^2 P^2 r + ... + gamma^(H-1) P^(H-1) r.
```

This expression is exact for a horizon of `H` steps.

## Transition-Reward Form

Because this project has rewards attached to individual transitions, it is often cleaner in code to avoid collapsing immediately to the expected reward vector. Instead compute:

```text
V_H(s) = sum_{s'} P(s' | s) * (reward(s, s') + gamma * V_{H-1}(s')).
```

This preserves the exact diagram specification directly.

## Q-Style View

There are no actions in an MRP, but a one-step lookahead quantity can still be defined in a Q-like way for each successor transition:

```text
Q_H(s, s') = R(s, s') + gamma V_{H-1}(s').
```

Then the Bellman recursion becomes:

```text
V_H(s) = sum_{s'} P(s' | s) Q_H(s, s').
```

This is not an MDP action-value function. It is simply a useful bookkeeping form for transition-level decomposition.

## Why This Exact Computation Is Mandatory Here

The state space is tiny, so exact finite-horizon values are cheap to compute and should be treated as ground truth for the chosen horizon and discount factor.

That gives two benefits:

1. Monte Carlo estimates can be checked against exact values.
2. Any mismatch beyond normal sampling noise will expose implementation bugs quickly.

## Relation To Monte Carlo

For a fixed `gamma`, start state `s`, and horizon `H`:

- Monte Carlo estimates `V_H(s)` by averaging sampled returns.
- Bellman recursion computes `V_H(s)` exactly from the transition probabilities and rewards.

As the number of trajectories increases, the Monte Carlo estimate should approach the exact finite-horizon value.

## Practical Algorithm

For a horizon `H = num_time_steps`:

1. Initialize `V = 0` for all states.
2. Repeat `H` times:
   - build a new vector `V_new`,
   - for each state `s`, compute

```text
V_new(s) = sum_{s'} P(s' | s) [ R(s, s') + gamma V(s') ].
```

3. Replace `V <- V_new`.
4. After `H` iterations, `V` equals the exact finite-horizon value vector.

This dynamic program is efficient and numerically stable for the small MRP in this project.
