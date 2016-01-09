---
layout: default
---

{% assign sorted_pages = site.pages | sort:"name" %}

# Chapters

{% for p in sorted_pages %}
    {% if p.layout == 'chapter' %}
        {% if p.status == 'stub' %}
- **{{ p.title }}**<br>{% else %}
- **<a class="chapter-link" href="{{ site.baseurl }}{{ p.url }}">{{ p.title }}</a>**<br>{% endif %}
    <em>{{ p.description }}</em>
    {% endif %}
{% endfor %}