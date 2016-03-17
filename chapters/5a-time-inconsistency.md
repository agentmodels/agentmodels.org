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
Rational, utility-maximizing agents are often modeled as *discounting* future utilities/rewards relative to present rewards. AI researchers construct systems for MDPs/RL with infinite time horizon that discount future rewards. Economists model humans or firms as discounting future rewards. Justifications for discounting include (a) avoiding problems with expected utilities diverging and (b) capturing human preference for the near-term (e.g. due to interest rates, vague deadlines, the chance of not being around in the future to realize gains).

Discounting in these examples is *exponential*. An exponential discounting agent appears to have some kind of inconsistency over time. With a discount rate of 0.8, $100 after 30 days is worth $0.12 and $110 at 31 days is $0.11 (assuming linear utility in money). Yet when the 30th day arrives, they are worth $100 and $88 respectively! (If insteaed the magnitudes were fixed from a starting time, the agent would have an overwhelming preference to travel back in time to get higher rewards!). Yet while these magnitudes have changed, the ratios stay fixed. Indeed, the ratios between any pair of outcomes are fixed regardless of the time the exponetial discounter evaluates them. So this agent thinks that two prospects in the far future are worthless compared to similar near-term prospects (disagreeing with their future self) but he agrees with his future self about which of the two worthless future prospects is better. [todo mention the relevance of this to planning in MDPs]. 

Any smooth discount function other than an exponential will result in preferences that reverse over time [cite]. So it's not so suprising that untutored humans should be subject to such reversals. (Without computational aids, human representations of numbers are systematically inaccurate. See refp:dehaene). Various functional forms for human discounting have been explored in the literature. We will describe the *hyperbolic discounting* model refp:ainslie2001breakdown because it is simple and well-studied. Any other functional form can easily be substituted into our models. 

The difference between hyperbolic and exponential discounting is illustrated in Figure 1. We plot the discount factor $$D$$ as a function of time $$t$$ in days. The exponential is $$D=\frac{1}{2^t}$$; the hyperbola is $$D=\frac{1}{1+2t}$$. These are not realistic discount rates. The important difference is that the hyperbola is initially steep and then becomes shallow (as the derivative rapidly decreases), while the exponetial is everywhere equally steep. 

![Figure 1](/assets/img/hyperbolic_no_label.jpg). 

Consider the example above but with different numbers. You are offered $100 after 4 days or $110 after 5 days. The discount factors for 4 and 5 days from the present are labeled in Figure 2. The change in $$D$$ from day 4 to 5 is small for the hyperbola (so waiting for $110 is preferred) and big for the exponential. When Day 4 arrives, you can get $100 immediately or $110 after 1 day. The difference between the curves is labeled on the left. The hyperbola is now steep and leads to you taking the $100 -- reversing your earlier preference.

**Exercise**: Calculate the discounted utilities for the two options ($100 vs. $110) for both hyperbolic and exponential discounting. First compute them when the $100 is 4 days from the present, then when it's 3 days from the present and so on (up to when it's 0 days from the present). 

![Figure 2](/assets/img/hyperbolic_label.jpg). 








