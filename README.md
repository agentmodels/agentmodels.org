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
npm install --save probmods/webppl webppl-timeit@latest webppl-dp@latest agentmodels/webppl-agents null-a/webppl-nn
cd node_modules/webppl
npm install
grunt bundle:../webppl-timeit:../webppl-dp:../webppl-agents:../webppl-nn
cp bundle/webppl.min.js ../../assets/js/webppl.min.js
cd ../..
~~~~
