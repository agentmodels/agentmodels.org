# Modeling Agents with Probabilistic Programs

## Development

Running a local server:

~~~~
jekyll serve
~~~~

## Updating dependencies

Once:

~~~~
npm install -g browserify uglifyjs watchify
~~~~

To update webppl and webppl packages (`./scripts/update-webppl`):

~~~~
npm install --save webppl@latest webppl-timeit@latest webppl-dp@latest probmods/webppl-viz agentmodels/webppl-gridworld
cd node_modules/vega
npm install
cd ../vega-lite
npm install
cd ../webppl
npm install
grunt compile:../webppl-timeit:../webppl-dp:../webppl-viz:../webppl-gridworld
cp compiled/webppl.min.js ../../assets/js/webppl.min.js
cd ../..
~~~~

To update the webppl editor (`./scripts/update-editor`):

~~~~
npm install --save probmods/webppl-editor
cd node_modules/wp-editor
npm install
make compiled/editor.js
cp compiled/editor.js ../../assets/js/webppl-editor.js
cp compiled/editor.css ../../assets/css/editor.css
cd ../..
~~~~