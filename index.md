---
layout: default
---

{% assign sorted_pages = site.pages | sort:"name" %}

{% for p in sorted_pages %}
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
        {% endif %} <!-- p.is_section -->
    {% endif %} <!-- p.layout == 'chapter' -->
{% endfor %}