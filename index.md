---
layout: default
title: "Modeling Agents with Probabilistic Programs"
---

<img src="/assets/img/maze.png" id="cover" />

## About this book

To be added. Placeholder: Formal models of rational agents play an important role in economics and in the cognitive sciences  as models of human or animal behavior. Core components of such models are expected-utility maximization, Bayesian inference, and game-theoretic equilibria. These ideas are also applied in engineering and in artificial intelligence in order to compute optimal solutions to problems and to construct artificial systems that learn and reason optimally. This tutorial implements utility-maximizing Bayesian agents as functional probabilistic programs. These programs provide a concise, intuitive translation of the mathematical specification of rational agents as code. The implemented agents explicitly simulate their own future choices via recursion. They update beliefs by exact or approximate Bayesian inference. They reason about other agents by simulating them (which includes simulating the simulations of others).

## Table of contents

{% assign sorted_pages = site.pages | sort:"name" %}

{% for p in sorted_pages %}
    {% if p.hidden %}
    {% else %}
        {% if p.layout == 'chapter' %}
            {% if p.is_section %}
                {% if p.status == 'stub' %}
1. **{{ p.title }}**<br>{% else %}
1. **<a class="chapter-link" href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a>**<br>{% endif %}
        <em>{{ p.description }}</em>
            {% else %}
                {% if p.status == 'stub' %}
    1. **{{ p.title }}**<br>{% else %}
    1. **<a class="chapter-link" href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a>**<br>{% endif %}
            <em>{{ p.description }}</em>        
            {% endif %}
        {% endif %}
    {% endif %}
{% endfor %}

## Citation

Please cite this book as: <br>
*Owain Evans, Andreas Stuhlmüller, John Salvatier, and Daniel Filan (electronic). Modeling Agents with Probabilistic Programs. Retrieved <span class="date"></span> from http://agentmodels.org.* <a id="toggle-bibtex" href="#" onClick="javascript:$('#bibtex').toggle();return false">[bibtex]</a>

<pre id="bibtex">
@misc{agentmodels,
  title = {% raw %}{{Modeling Agents with Probabilistic Programs}}{% endraw %},
  author = {Evans, Owain and Stuhlm\"{u}ller, Andreas and Salvatier, John and Filan, Daniel},
  year = {2016},
  howpublished = {\url{http://agentmodels.org}},
  note = {Accessed: <span class="date"></span>}
}
</pre>

## Open source

- [Book content](https://github.com/agentmodels/agentmodels.org)<br/>
  Markdown code for the book chapters
- [WebPPL](https://webppl.org)<br/>
  A probabilistic programming language for the web
- [WebPPL-Agents](https://github.com/agentmodels/webppl-agents)<br/>
  A library for modeling MDP and POMDP agents in WebPPL<br/>

## Acknowledgments

We thank Noah Goodman for helpful discussions, all WebPPL contributors for their work, and Long Ouyang for <a href="http://github.com/probmods/webppl-viz">webppl-viz</a>. This work was supported by Future of Life Institute grant 2015-144846 and by the Future of Humanity
Institute (Oxford).
