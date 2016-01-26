"use strict";


// Github links

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


// References and bibliography

var textohtml_map = {
  "\\\"u": "&uuml;",
  "\\\"a": "&auml;",
  "\\\"o": "&ouml;",
  "\\'e": "&eacute;",
  "\\\"U": "&Uuml;",
  "\\\"A": "&Auml;",
  "\\\"O": "&Ouml;",
  "\\'E": "&Eacute;",
  "\\\"{u}": "&uuml;",
  "\\\"{a}": "&auml;",
  "\\\"{o}": "&ouml;",
  "\\'{e}": "&eacute;",
  "\\\"{U}": "&Uuml;",
  "\\\"{A}": "&Auml;",
  "\\\"{O}": "&Ouml;",
  "\\'{E}": "&Eacute;"  
};

function textohtml(tex) {
    for (var key in textohtml_map) {
        if (textohtml_map.hasOwnProperty(key)) {
            tex = tex.replace("{" + key + "}", textohtml_map[key]);
            tex = tex.replace(key, textohtml_map[key]);
        };
    };
    return tex;
}

function replace_html(source, target) {
    $('p, li').each(function () {
        var html = $(this).html();
        $(this).html(html.replace(new RegExp(source, "ig"), target));
    });
}

function format_citation(citation) {
    var s = "";
    if (citation["URL"]) {
        s += "<a href='" + citation["URL"] + "'>" + citation["TITLE"] + "</a>. ";
    } else {
        s += citation["TITLE"] + ". ";
    };
    s += citation["AUTHOR"] + " (" + citation["YEAR"] + ").";
    if (citation["JOURNAL"]) {
        s += " <em>" + citation["JOURNAL"] + "</em>.";
    }
    return textohtml(s);
}

function format_reft(citation) {
  var s = "";
  if (citation["URL"]) {
    s += "<a href='" + citation["URL"] + "'>";
  }
  s += "<em>" + citation["AUTHOR"] + " (" + citation["YEAR"] + ")</em>";
  if (citation["URL"]) {
    s += "</a>";
  }  
  return textohtml(s);
}

function format_refp(citation) {
  var s = "(";
  if (citation["URL"]) {
    s += "<a href='" + citation["URL"] + "'>";
  }  
  s += "<em>" + citation["AUTHOR"] + "; " + citation["YEAR"] + "</em>";
  if (citation["URL"]) {
    s += "</a>";
  }
  s += ")";
  return textohtml(s);
}

$.get("/bibliography.bib", function (bibtext) {
    $(function () {
        var bibs = doParse(bibtext);
        $.each(
            bibs,
            function (citation_id, citation) {
              replace_html("cite:" + citation_id, format_citation(citation));
              replace_html("reft:" + citation_id, format_reft(citation));
              replace_html("refp:" + citation_id, format_refp(citation));
            }
        );
    });
});


// LaTeX math
// based on https://github.com/cben/sandbox/blob/gh-pages/_layouts/katex.html

$(function(){
  var scripts = document.getElementsByTagName("script");
  for (var i = 0; i < scripts.length; i++) {
    /* TODO: keep going after an individual parse error. */
    var script = scripts[i];
    if (script.type.match(/^math\/tex/)) {
      var text = script.text === "" ? script.innerHTML : script.text;
      var options = script.type.match(/mode\s*=\s*display/) ?
          {displayMode: true} : {};
      script.insertAdjacentHTML("beforebegin",
                                katex.renderToString(text, options));
    }
  }
  document.body.className += " math_finished";
});
