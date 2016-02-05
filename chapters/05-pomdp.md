---
layout: chapter
title: "Sequential decision problems with partial observability (POMDPs)"
description: Mathematical framework, implementation in WebPPL, Gridworld and restaurants example, bandit problems.
status: stub
---

Talk about different ways to represent belief states
- erp on erps as "full" solution
- various belief state approximations

~~~~
var drawGridWorld = function(world) {
  var element = wpEditor.makeResultContainer();
  GridWorld.draw(element, world);
}

var world = {
  xLim: 5,
  yLim: 5,
  blockedStates: [[0, 0], [0, 1]],
  terminals: [[2, 2], [2, 3]]
};

drawGridWorld(world);

var erp = Enumerate(function(){
    if (flip(.5)) {
       return "a";
    } else {
      if (flip(.5)) {
         return "b";
      } else {
         return "c";
      }
    }
});

viz.print(erp);
~~~~