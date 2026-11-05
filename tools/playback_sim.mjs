#!/usr/bin/env node
// 播放链路的无浏览器回归：真跑一次 /api/v1/tts/stream，再用真实的 pcm-worklet.js
// 和 player.js 在虚拟时钟上重放，检查「有没有丢样、播的时长对不对」。
//
// 为什么需要：v0.1.0 的卡顿就是主线程把整段 PCM 一股脑 postMessage 进 4 s 环形缓冲，
// 溢出的部分被 worklet 丢掉。浏览器里很难自动化，这里把音频线程的时钟虚拟化：
//   legacy  来一块推一块（旧逻辑，应当丢样）
//   gated   player.js 的信用窗口（新逻辑，不该丢样，且输出与输入逐样点相同）
//
// 用法：node tools/playback_sim.mjs [--exe csrc/build/moss_tts.exe] [--models models] [--port 8766]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const ROOT = path.resolve(import.meta.dirname, '..');
const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf('--' + name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};

const EXE = path.resolve(ROOT, arg('exe', 'csrc/build/moss_tts.exe'));
const MODELS = path.resolve(ROOT, arg('models', 'models'));
const PORT = Number(arg('port', 8766));
const BASE = `http://127.0.0.1:${PORT}`;
const TEXT = arg('text', '你好，我是 MOSS 语音助手。这段语音完全在你的电脑上合成，不需要联网，也不会把文字发到任何地方。');

const RATE = 48000;
const BLOCK = 128;
const TICK_MS = (BLOCK / RATE) * 1000;
const WORKLET_SRC = fs.readFileSync(path.join(ROOT, 'web/app/pcm-worklet.js'), 'utf8');
const PLAYER_SRC = fs.readFileSync(path.join(ROOT, 'web/app/player.js'), 'utf8');

/**
 * 在独立 vm 上下文里跑 pcm-worklet.js。
 * AudioWorklet 的 MessagePort 是「一对端点」：处理器侧的 port 发给主线程，
 * 主线程的 node.port 发给处理器。这里照实建两个端点，否则两边的 onmessage 会互相覆盖
 * （真实浏览器里不会，这是模拟器最容易踩的坑）。
 */
function loadWorklet(toMain) {
  let cls = null;
  const sandbox = {
    console,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = { onmessage: null, postMessage: (msg) => toMain.push(msg) };
      }
    },
    registerProcessor: (name, registered) => { cls = registered; },
  };
  vm.createContext(sandbox);
  vm.runInContext(WORKLET_SRC, sandbox, { filename: 'pcm-worklet.js' });
  if (!cls) throw new Error('pcm-worklet.js 没有注册处理器');
  return cls;
}

/** 在独立 vm 上下文里跑 player.js，返回 StreamPlayer。 */
function loadPlayer() {
  const sandbox = { console };
  vm.createContext(sandbox);
  vm.runInContext(PLAYER_SRC, sandbox, { filename: 'player.js' });
  if (!sandbox.StreamPlayer) throw new Error('player.js 没有导出 StreamPlayer');
  return sandbox.StreamPlayer;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const tick = () => new Promise((resolve) => setImmediate(resolve));

async function waitHealthy(child) {
  for (let i = 0; i < 150; ++i) {
    if (child.exitCode !== null) throw new Error('引擎提前退出：' + child.exitCode);
    try {
      const res = await fetch(BASE + '/api/v1/health');
      if (res.ok) return;
    } catch { /* 还没起来 */ }
    await sleep(200);
  }
  throw new Error('引擎没有起来');
}

/** 真跑一次流式接口，记录每个可用块的到达时刻（虚拟时间轴直接用它）。 */
async function recordStream() {
  const t0 = performance.now();
  const res = await fetch(BASE + '/api/v1/tts/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: TEXT, seed: 0, max_frames: 375 }),
  });
  if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + (await res.text()));
  const reader = res.body.getReader();
  const chunks = [];
  let pending = new Uint8Array(0);
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const all = new Uint8Array(pending.length + value.length);
    all.set(pending);
    all.set(value, pending.length);
    const usable = all.length - (all.length % 4);
    if (usable > 0) {
      const int16 = new Int16Array(all.slice(0, usable).buffer);
      const samples = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; ++i) samples[i] = int16[i] / 32768;
      chunks.push({ t: performance.now() - t0, samples });
    }
    pending = all.slice(usable);
  }
  return chunks;
}

/**
 * 虚拟时钟重放：每 128 帧调一次 process()，块按记录的时刻「到达」。
 * gated=true 用 player.js 的 send()（真代码）；false 模拟旧逻辑（来一块推一块）。
 */
async function replay(chunks, gated) {
  const toMain = [];
  const toWorklet = [];
  const worklet = new (loadWorklet(toMain))();
  const mainPort = { onmessage: null, postMessage: (msg) => toWorklet.push(msg) };
  const Player = gated ? loadPlayer() : null;
  const player = Player ? new Player({ sampleRate: RATE }, { port: mainPort }) : null;

  const block = () => [new Float32Array(BLOCK), new Float32Array(BLOCK)];
  const captured = [];
  const total = chunks.reduce((sum, c) => sum + c.samples.length / 2, 0);
  let capturedFrames = 0;

  // 真实浏览器里两个方向的 postMessage 都是异步任务，这里在 tick 边界统一投递
  const flushMessages = () => {
    while (toWorklet.length) {
      const msg = toWorklet.shift();
      if (worklet.port.onmessage) worklet.port.onmessage({ data: msg });
    }
    while (toMain.length) {
      const msg = toMain.shift();
      if (mainPort.onmessage) mainPort.onmessage({ data: msg });
    }
  };

  let clock = 0;
  let nextTick = TICK_MS;
  let index = 0;
  const runTick = () => {
    const out = block();
    const before = worklet.played;
    worklet.process([], [out]);
    const produced = worklet.played - before;
    if (produced > 0) captured.push([out[0].subarray(0, produced), out[1].subarray(0, produced)]);
    capturedFrames += produced;
    clock = nextTick;
    nextTick += TICK_MS;
    flushMessages();
  };

  /** 交货：gated 时可能因为信用窗口满而阻塞，阻塞期间必须继续推进时钟。 */
  async function deliver(chunk) {
    if (!gated) {
      worklet.onMessage({ type: 'push', samples: chunk.samples });
      return;
    }
    let settled = false;
    const pending = player.send(chunk.samples).then(() => { settled = true; });
    while (!settled) {
      if (nextTick <= clock) nextTick = clock + TICK_MS;
      runTick();
      await tick();
    }
    await pending;
  }

  while (index < chunks.length) {
    const arrive = chunks[index].t;
    if (arrive <= nextTick) {
      clock = arrive;
      await deliver(chunks[index]);
      index++;
    } else {
      runTick();
    }
  }
  if (gated) player.end(); else worklet.onMessage({ type: 'end' });
  flushMessages();
  for (let guard = 0; guard < 20000 && worklet.count > 0; ++guard) {
    if (nextTick <= clock) nextTick = clock + TICK_MS;
    runTick();
    flushMessages();
  }

  // 逐样点比对：gated 时播出来的必须就是「原样、不丢、不乱序」的那一段
  let maxDiff = 0;
  if (gated) {
    const expected = new Float32Array(total * 2);
    let offset = 0;
    for (const chunk of chunks) { expected.set(chunk.samples, offset); offset += chunk.samples.length; }
    let cursor = 0;
    for (const [left, right] of captured) {
      for (let i = 0; i < left.length && cursor < total; ++i, ++cursor) {
        maxDiff = Math.max(maxDiff, Math.abs(left[i] - expected[cursor * 2]));
        maxDiff = Math.max(maxDiff, Math.abs(right[i] - expected[cursor * 2 + 1]));
      }
      if (cursor >= total) break;
    }
  }

  return {
    totalFrames: total,
    playedFrames: worklet.played,
    droppedFrames: Math.floor(worklet.dropped / 2),
    underruns: worklet.underruns,
    capturedFrames,
    maxDiff,
  };
}

const sec = (frames) => (frames / RATE).toFixed(2);

async function main() {
  if (!fs.existsSync(EXE)) throw new Error('找不到 ' + EXE);
  const child = spawn(EXE, ['--models', MODELS, '--serve', '--no-open', '--port', String(PORT)], {
    stdio: ['ignore', 'ignore', 'inherit'],
  });
  try {
    await waitHealthy(child);
    const chunks = await recordStream();
    const total = chunks.reduce((sum, c) => sum + c.samples.length / 2, 0);
    const firstAt = chunks.length ? chunks[0].t : 0;
    const lastAt = chunks.length ? chunks[chunks.length - 1].t : 0;
    console.log(`[playback_sim] 音频 ${sec(total)} s · ${chunks.length} 个块 · 首块 ${Math.round(firstAt)} ms · 末块 ${Math.round(lastAt)} ms`);

    const legacy = await replay(chunks, false);
    const gated = await replay(chunks, true);
    const row = (name, r) =>
      `[${name.padEnd(6)}] 播放 ${sec(r.playedFrames)} s · 丢样 ${sec(r.droppedFrames)} s · 欠载 ${r.underruns} 次`;
    console.log(row('legacy', legacy) + '   ← 旧逻辑：来一块推一块');
    console.log(row('gated', gated) + '   ← player.js：信用窗口');
    if (gated.maxDiff !== 0) console.log(`[gated ] 输出与输入最大差异 ${gated.maxDiff}`);

    const problems = [];
    if (gated.droppedFrames !== 0) problems.push(`gated 丢了 ${sec(gated.droppedFrames)} s`);
    if (gated.playedFrames !== total) problems.push(`gated 只播了 ${sec(gated.playedFrames)} s / ${sec(total)} s`);
    if (gated.maxDiff !== 0) problems.push('gated 输出与输入的样点不一致');
    if (problems.length) {
      console.error('[fail] ' + problems.join('；'));
      process.exitCode = 1;
    } else {
      console.log('[ok] 信用窗口下逐样点一致，无丢样');
    }
  } finally {
    child.kill();
  }
}

main().catch((err) => {
  console.error('[fail] ' + (err && err.stack ? err.stack : err));
  process.exitCode = 1;
});
