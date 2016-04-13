(TeX-add-style-hook
 "figures"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "tikz"
    "amsmath"
    "amssymb")
   (LaTeX-add-labels
    "sec:procrastination-mdp"
    "sec:pomdp-model"
    "sec:irl-bandits"
    "sec:3c-first-todo"
    "sec:5c")))

