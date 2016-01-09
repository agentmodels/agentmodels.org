---
layout: default
---

{% assign sorted_pages = site.pages | sort:"name" %}

# Chapters

{% for p in sorted_pages %}
    {% if p.layout == 'chapter' %}
        {% if p.status == 'stub' %}
- **{{ p.title }}**<br>{% else %}
- **[{{ p.title }}]({{ site.baseurl }}{{ p.url }})**<br>{% endif %}
    <em>{{ p.description }}</em>
    {% endif %}
{% endfor %}