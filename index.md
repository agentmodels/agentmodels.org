---
layout: default
title: "Modeling Agents with Probabilistic Programs"
---

<img src="/assets/img/maze.png" id="cover" />

## About this book

The goal is to develop high-level models of rational agents for use in modeling human planning and inference. Agents are implemented as differentiable functional programs in a probabilistic programming language based on Javascript. Agents plan by recursively simulating their future selves or by simulating their opponents in multi-agent games. Our agents, environments and graph visualization all run in the browser -- no software needs to be installed. 

The book assumes basic programming experience but is otherwise self-contained. It includes short introductions to "planning as inference", MDPs, POMDPs, inverse reinforcement learning, hyperbolic discounting, myopic planning, and multi-agent planning.



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

We thank Noah Goodman for helpful discussions, all WebPPL contributors for their work, and Long Ouyang for <a href="http://github.com/probmods/webppl-viz">webppl-viz</a>. This work was supported by Future of Life Institute grant 2015-144846 and by the <a href="https://www.fhi.ox.ac.uk/">Future of Humanity
Institute</a> (Oxford).
