// Exercise the report script against the real embedded plot data without a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const page = fs.readFileSync(process.argv[2], 'utf8');
const script = fs.readFileSync(process.argv[3], 'utf8');
const data = JSON.parse(page.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);

const created = [];
const plotted = [];
const figures = ['combined', 'backend_wide', 'wide_layouts', 'explorer', 'row_scaling', 'profiles']
  .map(name => new Element('div', name));
const filters = new Element('div');
const handlers = {};

function Element(tag, figure) {
  this.tag = tag;
  this.figure = figure;
  this.children = [];
  this.options = [];
  this.value = '';
  this.selectedIndex = 0;
  this.listeners = {};
  this.classList = { add() {}, toggle() { return false; } };
}
Element.prototype.appendChild = function (child) {
  this.children.push(child);
  child.parentNode = this;
  if (this.tag === 'select') {
    this.options.push(child);
    if (this.options.length === 1) this.value = child.value;
  }
  return child;
};
Element.prototype.setAttribute = function (key, value) {
  if (key === 'text') this.textContent = value;
  else this[key] = value;
};
Element.prototype.getAttribute = function (key) { return key === 'data-figure' ? this.figure : null; };
Element.prototype.addEventListener = function (event, fn) { this.listeners[event] = fn; };

const document = {
  documentElement: {},
  getElementById(id) { return id === 'report-data' ? { textContent: JSON.stringify(data) } : filters; },
  querySelectorAll() { return figures; },
  createElement(tag) { const element = new Element(tag); created.push(element); return element; },
  createTextNode(text) { return { textContent: text }; }
};
const Plotly = { react(node, traces, layout) { plotted.push({ node, traces, layout }); } };
const window = {
  Plotly,
  addEventListener(event, fn) { handlers[event] = fn; },
  matchMedia() { return { addEventListener() {} }; }
};
vm.runInNewContext(script, { document, window, Plotly, getComputedStyle() {
  return { getPropertyValue() { return '#222222'; } };
} });
handlers.load();

function assertTitles() {
  assert.ok(plotted.length > 0);
  for (const { layout } of plotted) {
    if (layout.yaxis.type === 'log') {
      assert.match(layout.yaxis.title.text, /\(log scale\)$/);
    } else if (layout.yaxis.title) {
      assert.doesNotMatch(layout.yaxis.title.text, /\(log scale\)$/);
    }
  }
}
assert.ok(plotted.some(({ layout }) => layout.yaxis.type === 'log'));
assertTitles();

const logBox = created.find(node => node.tag === 'input' && node.id === 'log-y');
assert.ok(logBox);
logBox.checked = false;
plotted.length = 0;
logBox.listeners.change();
// Overview and explorer switch to linear. Row scaling and profiles stay logarithmic.
assertTitles();
assert.ok(plotted.some(({ layout }) => layout.yaxis.type === 'linear' && layout.yaxis.title));
assert.ok(plotted.some(({ layout }) => layout.yaxis.type === 'log'));

logBox.checked = true;
logBox.listeners.change();
const operation = created.find(node => node.tag === 'select' && node.options.some(option => option.value === 'storage_size'));
assert.ok(operation);
operation.value = 'storage_size';
plotted.length = 0;
operation.listeners.change();
assertTitles();
assert.ok(plotted.some(({ layout }) => layout.yaxis.title?.text === 'Storage size (bytes) (log scale)'));

// Empty panels use a linear placeholder axis and must not claim a log scale.
const backendButtons = created.filter(node => node.tag === 'button' && node.listeners.click &&
  data.backends.some(backend => backend.label === node.textContent));
assert.equal(backendButtons.length, data.backends.length);
backendButtons.forEach(node => node.listeners.click());
plotted.length = 0;
logBox.listeners.change();
assertTitles();
assert.ok(plotted.every(({ layout }) => layout.yaxis.type !== 'log'));
