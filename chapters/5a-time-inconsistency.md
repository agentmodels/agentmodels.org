---
layout: chapter
title: "Time inconsistency"
description: Hyperbolic discounting, "donut temptation" example, procrastination example (from NIPS workshop paper).

---

## PLAN

- Intuitive examples. Setting alarm clock. Procrastinating on giving comments on paper. 

- General phenomenon (explored by experiments). Example from paper / slide. Show two curves. Graphic from Frank presentation. Hyperbola special? No, any thing non-exponential time inconsistent. Various functional forms explored in literature with different analytic and computational properties. Could easily plug other forms into our models.

- Naive vs. Sophisticated. For sequential planning, issue of whether agent takes bias into account (can't directly control future actions but can take actions now that preclude states in which bad actions would occur). 

- Formal model: Add delays to MDP model (and similarly for POMDP model). Depending on delay you put into simulated call to act(), get Naive or Sophisticated. 

