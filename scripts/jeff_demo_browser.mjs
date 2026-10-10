// Headless check of scripts/jeff_demo.html against a running system1-ane-serve, through the installed Google Chrome.
// Needs playwright-core (no browser download): npm i playwright-core   (in any folder; pass it via NODE_PATH)
//   NODE_PATH=/path/to/node_modules node scripts/jeff_demo_browser.mjs [--url http://127.0.0.1:8787/] [--out dir]
//     [--html page.html]   (serve this file as the page instead of the server's copy)
// Checks: Snake target arrow and Move text; Tetris mid-game Reset with a request in flight; Pause / Play;
// top-out before a request and right after a placement, with the translucent GAME OVER overlay; Play starts a
// new game. Exits nonzero on the first failed check.
import { createRequire } from "node:module";
import { mkdirSync, readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright-core");

const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const url = arg("--url", "http://127.0.0.1:8787/");
const out = arg("--out", "/tmp/jeff_demo_browser");
mkdirSync(out, { recursive: true });

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!ok) throw new Error(`check failed: ${name}`);
}

const html = arg("--html", null);
const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1180, height: 1400 } });
if (html) {
  // Serve this file as the page (the API still goes to --url), to test page edits without touching the server.
  await page.route((u) => u.pathname === "/", (route) => route.fulfill({ contentType: "text/html", body: readFileSync(html, "utf8") }));
}
page.on("pageerror", (error) => console.log("page error:", error.message));
const state = () => page.evaluate(() => ({
  pieces: tPieces, lines: tLines, playing: tPlaying, inflight: tInflight, generation: tGeneration, over: tOver,
  filled: tBoard.flat().filter(Boolean).length, piece: tPiece, next: tNext,
  overlay: !document.getElementById("tetris-over").hidden,
  why: document.getElementById("tetris-over-why").textContent,
}));
const waitFor = (fn, arg, timeout = 30000) => page.waitForFunction(fn, arg, { timeout, polling: 20 });

// Play after game over: a fresh board (new generation, no stats, overlay hidden) that starts dropping pieces.
async function playAfterGameOver(name) {
  const over = await state();
  if (!over.over || !over.overlay) throw new Error(`${name}: not at game over`);
  await page.click("#tetris-toggle");
  const fresh = await state();
  await waitFor(() => tPieces > 0, undefined, 5000).catch(() => {});
  const later = await state();
  check(name, fresh.generation > over.generation && !fresh.overlay && !fresh.over && fresh.playing
    && fresh.filled === 0 && fresh.pieces === 0 && fresh.lines === 0
    && later.pieces > 0 && !later.overlay && later.filled <= 4 * later.pieces,
    `at click ${JSON.stringify(fresh)}; within 5 s pieces ${later.pieces}, filled ${later.filled}`);
}

try {
  await page.goto(url);
  await waitFor(() => document.getElementById("health").textContent.includes("ready"));

  // Snake: the chosen cell gets the cyan outline and an arrow; the readout names the move.
  await waitFor(() => document.querySelector("#board .cell.target .arrow"));
  const snake = await page.evaluate(() => {
    const cell = document.querySelector("#board .cell.target");
    const arrow = cell.querySelector(".arrow");
    const cs = getComputedStyle(cell);
    return {
      arrow: arrow.textContent, color: getComputedStyle(arrow).color, outline: cs.outlineColor + " " + cs.outlineWidth,
      move: document.getElementById("move").textContent, chosenFill: getComputedStyle(document.querySelector("#bars .bar.chosen .fill")).backgroundColor,
      otherFill: [...document.querySelectorAll("#bars .bar:not(.chosen) .fill")].map((f) => getComputedStyle(f).backgroundColor)[0],
      head: getComputedStyle(document.querySelector("#board .cell.h")).backgroundColor,
    };
  });
  await page.locator("#game").screenshot({ path: `${out}/snake_target.png` });
  check("snake target arrow + Move text", /^Move: (UP|DOWN|LEFT|RIGHT)$/.test(snake.move) && snake.color === "rgb(0, 229, 255)"
    && snake.outline === "rgb(0, 229, 255) 3px", JSON.stringify(snake));
  check("probability bars: chosen cyan, others grey", snake.chosenFill === "rgb(0, 229, 255)" && snake.otherFill === "rgb(92, 99, 112)");

  // A target on the food: the arrow sits over the red dot.
  await page.click("#toggle");
  await waitFor(() => inflight === false);
  const onFood = await page.evaluate(() => {
    target = { r: food.r, c: food.c, dir: "left" };
    draw();
    const cell = document.querySelector("#board .cell.target");
    return { food: cell.classList.contains("f"), dot: getComputedStyle(cell, "::before").backgroundColor,
      arrow: cell.querySelector(".arrow").textContent, arrowPx: cell.querySelector(".arrow").getBoundingClientRect().height,
      cellPx: cell.getBoundingClientRect().height };
  });
  await page.locator("#board").screenshot({ path: `${out}/snake_target_on_food.png` });
  check("target on food: arrow over the red dot", onFood.food && onFood.dot === "rgb(255, 23, 68)" && onFood.arrow.startsWith("\u25C0"),
    JSON.stringify(onFood));
  await page.evaluate(() => { target = null; draw(); });
  await page.click("#toggle");

  // Snake Reset with a request in flight: the stale answer must not move the fresh snake.
  await waitFor(() => inflight === true);
  await page.click("#reset");
  const afterReset = await page.evaluate(() => ({ head: snake[0], len: snake.length, target, move: document.getElementById("move").textContent }));
  check("snake reset clears target", afterReset.target === null && afterReset.move === "Move: —" && afterReset.len === 3,
    JSON.stringify(afterReset));

  const cachedLines = await page.evaluate(() => {
    const saved = [...document.querySelectorAll("#tetris-timing span")].map((s) => s.textContent);
    showTiming("tetris-", {
      usage: { input_tokens: 821 },
      timings: { tokenize_ms: 1, calls_ms: [40, 11], call_widths: [512, 256], prefill_ms: 51, head_ms: 0.5,
                 total_ms: 53, questions: [{ prefix_tokens: 136 }] },
    }, "tetris");
    const lines = [...document.querySelectorAll("#tetris-timing span")].map((s) => s.textContent);
    document.querySelectorAll("#tetris-timing span").forEach((s, i) => { s.textContent = saved[i]; });
    return lines;
  });
  check("timing lines: prefilled tokens exclude the prefix cache", cachedLines[0] === "tetris · 821 tokens · 136 from the prefix cache"
    && cachedLines[1] === "tokenize 1.0 · prefill 685 tok (512 + 256) 51.0 ms · head 0.5 · server other 0.5 ms", JSON.stringify(cachedLines));

  // Tetris: play, then Reset mid-game while a request is in flight.
  await page.click("#tetris-toggle");
  await waitFor(() => tPieces >= 3);
  await waitFor(() => tInflight === true);
  const before = await state();
  const stagesText = () => page.evaluate(() => [...document.querySelectorAll("#tetris-stages span")].map((s) => s.textContent));
  const stages = await stagesText();
  const num = (text) => Number.parseFloat(text);
  const [, tok, pre, head, other, wall] = stages.slice(6, 12);
  check("stage table: last move and game average", stages.length === 18 && stages[6] === "last"
    && /^avg \d+$/.test(stages[12]) && /^\d+(\.\d)? \(\d+\)$/.test(pre)
    && Math.abs(num(tok) + num(pre) + num(head) + num(other) - num(wall)) <= 2.5, JSON.stringify(stages));
  await page.click("#tetris-reset");
  const reset0 = await state();
  await page.waitForTimeout(2500);
  const reset1 = await state();
  check("reset clears the stage table and average", (await stagesText()).length === 0
    && (await page.evaluate(() => document.getElementById("tetris-avg").textContent)) === "");
  check("mid-game reset with a request in flight", before.inflight && before.pieces >= 3 && !reset0.playing
    && reset0.generation > before.generation && reset1.pieces === 0 && reset1.lines === 0 && reset1.filled === 0
    && !reset1.playing && !reset1.overlay, `before ${JSON.stringify(before)} after 2.5 s ${JSON.stringify(reset1)}`);

  // Pause / Play.
  await page.click("#tetris-toggle");
  await waitFor(() => tPieces >= 2);
  await waitFor(() => tInflight === true);
  await page.click("#tetris-toggle");
  await waitFor(() => tInflight === false);
  const paused0 = await state();
  await page.waitForTimeout(1500);
  const paused1 = await state();
  check("pause stops after the in-flight move", !paused1.playing && paused1.pieces === paused0.pieces,
    `${paused0.pieces} -> ${paused1.pieces}`);
  await page.click("#tetris-toggle");
  await waitFor((n) => tPieces >= n + 2, paused1.pieces);
  check("play resumes the same game", (await state()).pieces >= paused1.pieces + 2);
  await page.click("#tetris-toggle");
  await waitFor(() => tInflight === false);

  // Top-out before a request: the spawn cells of the current piece are already filled.
  await page.evaluate(() => {
    tBoard = tEmpty();
    for (let r = 1; r < T_H; r += 1) for (let c = 1; c < T_W; c += 1) tBoard[r][c] = "Z";
    tPiece = "T";
    tDraw();
  });
  await page.click("#tetris-toggle");
  await waitFor(() => tOver === true, undefined, 5000);
  const top1 = await state();
  check("top-out before the request", top1.overlay && !top1.playing && top1.why.startsWith("Topped out · Lines"), top1.why);
  const log1 = await page.evaluate(() => document.getElementById("tetris-status").firstChild.textContent);
  check("game over is logged with lines and pieces", /^Game over: Topped out\. Lines \d+, pieces \d+\.$/.test(log1), log1);

  // Overlay: translucent, exactly over the board, board visible underneath.
  const overlay = await page.evaluate(() => {
    const o = document.getElementById("tetris-over").getBoundingClientRect();
    const b = document.getElementById("tetris-board").getBoundingClientRect();
    const cs = getComputedStyle(document.getElementById("tetris-over"));
    const title = getComputedStyle(document.querySelector("#tetris-over strong"));
    return { o: [o.x, o.y, o.width, o.height], b: [b.x, b.y, b.width, b.height], bg: cs.backgroundColor,
      blur: cs.backdropFilter || cs.webkitBackdropFilter, color: title.color, weight: title.fontWeight, size: title.fontSize,
      shadow: title.textShadow };
  });
  await page.locator(".tboard-wrap").screenshot({ path: `${out}/tetris_game_over.png` });
  check("overlay covers exactly the board", JSON.stringify(overlay.o) === JSON.stringify(overlay.b), JSON.stringify(overlay));
  check("overlay is translucent", overlay.bg === "rgba(0, 0, 0, 0.45)" && overlay.color === "rgb(255, 255, 255)"
    && Number(overlay.weight) >= 700 && overlay.shadow !== "none", `${overlay.bg} ${overlay.blur}`);

  await playAfterGameOver("play after a top-out before the request");
  await page.click("#tetris-toggle");
  await waitFor(() => tInflight === false);

  // Top-out right after a placement: only the two top rows are open, so a piece lands under the spawn area.
  await page.evaluate(() => {
    tBoard = tEmpty();
    for (let r = 2; r < T_H; r += 1) for (let c = 0; c < T_W; c += 1) tBoard[r][c] = (r === 2 ? c < T_W - 1 : c > 0) ? "L" : 0;
    tPieces = 0;
    tLines = 0;
    tDraw();
  });
  await page.click("#tetris-toggle");
  await waitFor(() => tOver === true, undefined, 60000);
  const top2 = await state();
  const log2 = await page.evaluate(() => [...document.getElementById("tetris-status").children].map((p) => p.textContent));
  check("top-out after a placement", top2.overlay && !top2.playing && top2.pieces >= 1 && top2.why.startsWith("Topped out"),
    `${top2.why} | ${log2.slice(0, 2).join(" | ")}`);
  await page.waitForTimeout(800);
  check("no request after game over", (await state()).pieces === top2.pieces && !(await state()).inflight);
  await page.locator(".tboard-wrap").screenshot({ path: `${out}/tetris_game_over_after_placement.png` });
  await playAfterGameOver("play after a top-out after a placement");
  await page.click("#tetris-toggle");
  await waitFor(() => tInflight === false);

  // Reset hides the overlay.
  await page.click("#tetris-reset");
  const cleared = await state();
  check("reset hides the overlay", !cleared.overlay && cleared.filled === 0 && cleared.pieces === 0);
  await page.screenshot({ path: `${out}/page.png`, fullPage: true });
} finally {
  await browser.close();
}
console.log(`${results.filter((r) => r.ok).length}/${results.length} checks passed; screenshots in ${out}`);
