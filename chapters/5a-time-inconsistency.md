---
layout: chapter
title: "Time inconsistency"
description: Hyperbolic discounting, "donut temptation" example, procrastination example (from NIPS workshop paper).

---

## PLAN

- Intuitive examples. Setting alarm clock. Procrastinating on giving comments on paper. 

- General phenomenon (explored by experiments). Example from paper / slide. Show two curves. Graphic from Frank presentation. Hyperbola special? No, any thing non-exponential time inconsistent. Various functional forms explored in literature with different analytic and computational properties. Could easily plug other forms into our models.

- Naive vs. Sophisticated. For sequential planning, issue of whether agent takes bias into account (can't directly control future actions but can take actions now that preclude states in which bad actions would occur). 

- Formal model: Add delays to MDP model (and similarly for POMDP model). Depending on delay you put into simulated call to act(), get Naive or Sophisticated. Delays used to implement other time inconsistent agents. 

- Implementation. Gridworld examples

### Introduction
Time inconsistency is part of everyday human experience. The night before you wish to rise early; in the morning you prefer to sleep in. There is an inconsistency between what you prefer your future self to do and what your future self prefers to do. Forseeing this inconsistency, you take actions the night before to bind your future self to getting up. These range from setting an alarm clock to arranging for someone to drag you out of bed.

Similar examples abound. People pay upfront for gym subscriptions they rarely use. People procrastinate on writing papers: they plan to start the paper early but then delay until the last minute. The practical consequences of time inconsistency are substantial in different domains [ref: highbrow gathering dust, stanford-cornell mooc paper with procommitment intervention, arizona?/NM birth control intervention]. 

Time inconsistency has been used to explain not just quotidian laziness but also addiction, procrastination, impulsive behavior as well an array of "pre-commitment" behaviors :refp:ainslie2001breakdown. Lab experiments of time inconsistency often use simple quantitative questions such as:

<blockquote>Would you prefer to get $100 after 30 days or $110 after 31 days?
</blockquote>

Most people answer "yes" to this question. But their preference reverses once the 30th day comes around and they contemplate getting $100 immediately. The next section describes a formal models that predicts this reversal. This model is then incorporated into our model for MDP planning and implemented in WebPPL. 

### Time inconsistency due to hyperbolic discounting
Rational, utility-maximizing agents are often modeled as *discounting* future utilities/rewards relative to present rewards. AI researchers construct systems for MDPs/RL with infinite time horizon that discount future rewards. Economists model humans or firms as discounting future rewards. Justifications for discounting include (a) avoiding purely mathematical problems with infinities, and (b) capturing human preference for the near-term (e.g. due to chance of not being around in the future to realize gains).

Discounting in these examples is *exponential*. An exponential discounting agent has a distinct air of time inconsistency. With a discount rate of 0.8, $100 after 30 days is worth $0.12 and $110 at 31 days is $0.11. Yet when the 30th day arrives, they are worth $100 and $88 respectively. It certainly seems like the agent's preferences have changed over time, even if this particular preference ordering hasn't reversed. 


