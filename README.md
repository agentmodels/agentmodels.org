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

To update webppl:

~~~~
npm install --save webppl@latest
cd node_modules/webppl
npm install
grunt compile:../webppl-timeit:../webppl-dp:../webppl-viz
cp compiled/webppl.min.js ../../assets/js/webppl.min.js
cd ../..
~~~~

To update the webppl editor:

~~~~
npm install --save probmods/webppl-editor
cd node_modules/wp-editor
npm install
make compiled/editor.js
cp compiled/editor.js ../../assets/js/webppl-editor.js
cp compiled/*.css ../../assets/css/
cd ../..
~~~~