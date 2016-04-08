# Modeling Agents with Probabilistic Programs

## Setup

~~~~
npm install -g browserify uglifyjs watchify grunt-cli
gem install jekyll
~~~~

## Development

Running a local server:

~~~~
jekyll serve
~~~~

## Updating dependencies

To update webppl and webppl packages (`./scripts/update-webppl`):

~~~~
npm install --save webppl@latest webppl-timeit@latest webppl-dp@latest agentmodels/webppl-gridworld
cd node_modules/webppl
npm install
grunt bundle:../webppl-timeit:../webppl-dp:../webppl-gridworld
cp bundle/webppl.min.js ../../assets/js/webppl.min.js
cd ../..
~~~~

To update webppl-viz (`./scripts/update-viz`):

~~~~
npm install --save probmods/webppl-viz
cd node_modules/vega
npm install
cd ../vega-lite
npm install
cd ../webppl-viz
make demo/webppl-viz.js
cp demo/webppl-viz.js ../../assets/js/webppl-viz.js
cp demo/webppl-viz.css ../../assets/css/webppl-viz.css
cd ../..
~~~~

If you get the error "cd: no such file or directory: vega" update npm.

~~~~
npm -g install npm
~~~~

To update the webppl editor (`./scripts/update-editor`):

~~~~
npm install --save probmods/webppl-editor
cd node_modules/wp-editor
npm install
make all
cp compiled/editor.js ../../assets/js/webppl-editor.js
cp compiled/editor.css ../../assets/css/editor.css
cd ../..
~~~~
