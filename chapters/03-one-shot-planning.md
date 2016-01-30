---
layout: chapter
title: One-shot planning
description: Various agent models for solving one-shot decision problems. 
---

Start with one-shot planning. Choices have some consequence. We take action that has best consequences. Implement two ways.

Outcomes probabilistic: lotteries, games of chance. Take action that is best in expectation. Write down equation.

Planning as inference for softmax. Write down softmax. Not obvious why it's a normative model. But clear that it could be good model of humans or other agents. x

## Agents for simple decision problems

The goal for the next two chapters is to build up to agent models that solve decision problems that involve long sequences of actions (e.g. MDPs). We start with "one-shot" decision problems where the agent selects a single action. These problems are trivial to solve without WebPPL. The point is to illustrate the WebPPL idioms we'll use to tackle more complex problems. 

## One-shot decisions: deterministic actions
In a decision problem, an agent must choose between a set of actions. The agent will try to choose the action that is best in terms of their own preferences. This usually depends only on the *consequences* of the action. So the agent will try to pick actions with preferable consequences.

For example, suppose Tom is choosing between restaurants and he cares only about getting pizza. There's an Italian restaurant and a French restaurant. So John will choose the Italian restaurant because it leads to the state where gets pizza.

Formally, Tom selects an action $$a$$ from the set of actions $$A$$, which includes the two restaurants. The consequences of an action are represented by a transition function $$T:(S,A) \to S$$ from state-action pairs to states. In our example, the relevant states are whether or not Tom gets pizza. Tom's preferences are represented by a utility function $$U:(S) \to \mathbb{R}$$, which indicates the relative goodness of each state. 

Tom's decision rule is to take action $$a$$ such that:
$$
[\max_{a \in A} U(T(s,a))]
$$




