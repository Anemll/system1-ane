"""Exercise the demo's health failure and single-click recovery in Node."""
from pathlib import Path
import shutil
import subprocess
import unittest


DEMO = Path(__file__).resolve().parents[1]/'scripts'/'jeff_demo.html'
NODE_TEST = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
assert.match(source, /let playing = false;/);
assert.match(source, /<button id="toggle" type="button">Play<\/button>/);
assert.match(source, /<select id="adapter"[^>]*disabled>/);
const handlerStart = source.indexOf('document.getElementById("toggle").addEventListener("click", async () => {');
const handlerEnd = source.indexOf('\n});', handlerStart) + 4;
const healthStart = source.indexOf('async function loadHealth() {');
const healthEnd = source.indexOf('\n}\n\nfor (const button of document.querySelectorAll(".modes button")', healthStart) + 2;
assert.ok(handlerStart > 0 && handlerEnd > handlerStart && healthStart > 0 && healthEnd > healthStart);
const elements = new Proxy({}, {get(target, key) {
  return target[key] ||= {textContent: '', disabled: false, value: '',
    addEventListener(_name, callback) {this.click = callback;}};
}});
let attempts = 0, steps = 0;
elements.adapter.disabled = true;
const context = vm.createContext({
  document: {getElementById: id => elements[id]},
  fetch: async () => {
    if (++attempts === 1) throw new Error('offline');
    return {ok: true, json: async () => ({status: 'ready', model: 'Jeff', backend: 'ANE',
      prefix_cache: false, adapters: ['base', 'snake']})};
  },
  URLSearchParams, location: {search: ''},
  fillAdapters() {}, showSample() {}, tStop() {}, step() {steps++;}, logStatus() {},
});
vm.runInContext('let playing = false, healthReady = false, healthLoading = false, adapter = "base";\n' +
  source.slice(handlerStart, handlerEnd) + '\n' + source.slice(healthStart, healthEnd), context);
(async () => {
  await vm.runInContext('loadHealth()', context);
  assert.equal(vm.runInContext('playing', context), false);
  assert.equal(elements.toggle.textContent, 'Retry');
  assert.match(elements.health.textContent, /Health check failed.*Click Retry/);
  assert.equal(steps, 0);
  assert.equal(elements.adapter.disabled, true);
  await elements.toggle.click();
  assert.equal(vm.runInContext('playing', context), true);
  assert.equal(vm.runInContext('healthReady', context), true);
  assert.equal(vm.runInContext('adapter', context), 'snake');
  assert.equal(elements.toggle.textContent, 'Pause');
  assert.equal(steps, 1);
  assert.equal(elements.adapter.disabled, false);
  // Snake/text selector must not list Tetris adapters; the Tetris selector keeps base + tetris*.
  const fills = [];
  const context2 = vm.createContext({
    document: {getElementById: id => elements[id]},
    fetch: async () => ({ok: true, json: async () => ({status: 'ready', model: 'Jeff', backend: 'ANE',
      prefix_cache: false, adapters: ['base', 'snake', 'snake-stock', 'tetris', 'tetris-previous', 'triage']})}),
    URLSearchParams, location: {search: ''},
    fillAdapters(select, names) {fills.push([select, names]);},
    showSample() {}, tStop() {}, tStart() {}, step() {}, logStatus() {},
  });
  vm.runInContext('let playing = false, healthReady = false, healthLoading = false, adapter = "base";\n' +
    source.slice(handlerStart, handlerEnd) + '\n' + source.slice(healthStart, healthEnd), context2);
  await vm.runInContext('loadHealth()', context2);
  assert.equal(fills.length, 2);
  assert.equal(fills[0][0], elements.adapter);
  assert.deepEqual(fills[0][1], ['base', 'snake', 'snake-stock', 'triage']);
  assert.equal(fills[1][0], elements['tetris-adapter']);
  assert.deepEqual(fills[1][1], ['base', 'tetris', 'tetris-previous']);
  assert.equal(vm.runInContext('adapter', context2), 'snake');
})().catch(error => {console.error(error); process.exitCode = 1;});
"""


@unittest.skipUnless(shutil.which('node'), 'Node is needed to run demo JavaScript')
class DemoHealthTests(unittest.TestCase):
    def test_failed_health_then_one_click_retries_and_starts(self):
        result = subprocess.run(['node', '-e', NODE_TEST, str(DEMO)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
