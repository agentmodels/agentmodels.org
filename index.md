---
layout: default
title: "Modeling Agents with Probabilistic Programs"
---

<img src="/assets/img/maze.png" id="cover" />

### About this book

This book describes and implements models of rational agents for (PO)MDPs and Reinforcement Learning. One motivation is to create richer models of human planning, which capture human biases and bounded rationality. 

Agents are implemented as differentiable functional programs in a probabilistic programming language based on Javascript. Agents plan by recursively simulating their future selves or by simulating their opponents in multi-agent games. Our agents and environments run directly in the browser and are easy to modify and extend.

The book assumes basic programming experience but is otherwise self-contained. It includes short introductions to <a href="/chapters/3-agents-as-programs.html#planning_as">"planning as inference"</a>, [MDPs](/chapters/3a-mdp.html), [POMDPs](/chapters/3c-pomdp.html), [inverse reinforcement learning](/chapters/4-reasoning-about-agents.html), [hyperbolic discounting](/chapters/5a-time-inconsistency.html), [myopic planning](/chapters/5c-myopic.html), and [multi-agent planning](/chapters/multi-agent.html).

For more information about this project, contact [Owain Evans](http://owainevans.github.io). 



### Table of contents

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

### Citation

Please cite this book as: <br>
Owain Evans, Andreas Stuhlmüller, John Salvatier, and Daniel Filan (electronic). *Modeling Agents with Probabilistic Programs.* Retrieved <span class="date"></span> from `http://agentmodels.org`. <a id="toggle-bibtex" href="#" onClick="javascript:$('#bibtex').toggle();return false">[bibtex]</a>

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
