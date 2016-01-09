"use strict";

var github_repository = "https://github.com/agentmodels/agentmodels.org/";

function markdown_url(page_url) {
    return page_url.slice(0, -4) + "md";
}

function github_edit_url(page_url) {
    return github_repository + "edit/gh-pages" + markdown_url(page_url);
}

function github_page_url(page_url) {
    if ((page_url == "/index.html") || (page_url == "/")) {
        return github_repository + "blob/gh-pages/chapters";
    } else {
        return github_repository + "blob/gh-pages" + markdown_url(page_url);
    };
}
