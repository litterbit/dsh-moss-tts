#!/usr/bin/env node
// 克隆链路的无浏览器回归：真实引擎 + 真实的 web/app/clone-audio.js。
//
// 为什么需要：网页现在在浏览器里解码 MP3/M4A，再按用户拖出来的 20 秒窗口重新打包 PCM WAV
// 交给引擎。这条链路最容易坏在两处 —— 打包出来的 WAV 头不对（引擎报"块对齐不符"），
// 或者超长/压缩格式悄悄被接受。这里把这两头都钉死：
//   1) clone-audio.js 的纯逻辑（选段、波形、WAV 打包）。
//   2) index.html 与 app.js 的元素 id 契约（少了 id 只会在浏览器里静默报错）。
//   3) 真跑 moss_tts --serve：12 秒整段 / 8 秒窗口 / 25 秒超长 / MP3-in-WAV，四种都要对。
//
// 用法：node tools/clone_sim.mjs [--exe csrc/build/moss_tts.exe] [--models models] [--port 8793]
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
const PORT = Number(arg('port', 8793));
const BASE = `http://127.0.0.1:${PORT}`;
const RATE = 48000;
const WORK = path.join(ROOT, 'runs', 'clone-sim');

let failures = 0;
function check(name, ok, detail = '') {
  if (ok) {
    console.log(`  [ok] ${name}`);
  } else {
    failures++;
    console.log(`  [x] ${name}${detail ? ' — ' + detail : ''}`);
  }
}

// ---------------------------------------------------------------- 纯逻辑
const CLONE_SRC = fs.readFileSync(path.join(ROOT, 'web/app/clone-audio.js'), 'utf8');
const sandbox = { console };
vm.createContext(sandbox);
vm.runInContext(CLONE_SRC, sandbox, { filename: 'clone-audio.js' });
const Trim = sandbox.MossCloneAudio;
if (!Trim) throw new Error('clone-audio.js 没有导出 MossCloneAudio');

console.log('[1] clone-audio.js');
check('上限是 20 秒', Trim.MAX_SECONDS === 20);
{
  const w = Trim.defaultWindow(35);
  check('长音频默认取前 20 秒', w.start === 0 && w.end === 20, JSON.stringify(w));
  const short = Trim.defaultWindow(8);
  check('短音频默认整段', short.start === 0 && short.end === 8, JSON.stringify(short));
}
{
  // 顶到上限后继续拖终点：窗口整段往后滑（长音频选后段就是这么操作）
  const w = Trim.dragWindow(35, { start: 0, end: 20 }, 'end', 35);
  check('拖终点到 35 s → 15~35 s', Math.abs(w.start - 15) < 1e-9 && Math.abs(w.end - 35) < 1e-9, JSON.stringify(w));
  const shrunk = Trim.dragWindow(35, { start: 0, end: 20 }, 'start', 5);
  check('拖起点 = 从左边裁', Math.abs(shrunk.start - 5) < 1e-9 && Math.abs(shrunk.end - 20) < 1e-9,
        JSON.stringify(shrunk));
  const clamped = Trim.dragWindow(35, { start: 0, end: 20 }, 'start', 30);
  check('起点不能越过终点', clamped.end - clamped.start >= Trim.MIN_SECONDS - 1e-9, JSON.stringify(clamped));
  const beyond = Trim.dragWindow(8, { start: 0, end: 8 }, 'end', 99);
  check('终点不超过文件长度', Math.abs(beyond.end - 8) < 1e-9, JSON.stringify(beyond));
  const fromMiddle = Trim.moveWindow(35, { start: 0, end: 20 }, 30);
  check('整段平移夹在边界内', Math.abs(fromMiddle.start - 15) < 1e-9, JSON.stringify(fromMiddle));
}
{
  const left = new Float32Array(RATE * 2);
  const right = new Float32Array(RATE * 2);
  left[RATE] = 0.5;   // 1.0 s 处的标记样点
  const [l, r] = Trim.sliceWindow([left, right], { start: 1, end: 2 });
  check('sliceWindow 长度 = 1 s', l.length === RATE && r.length === RATE, `${l.length}`);
  check('sliceWindow 从起点开始', Math.abs(l[0] - 0.5) < 1e-9, `${l[0]}`);
}
{
  const peaks = Trim.peaks([new Float32Array([0, 1, -1, 0.25])], 2);
  check('peaks 每列取绝对值最大', Math.abs(peaks[0] - 1) < 1e-6 && Math.abs(peaks[1] - 1) < 1e-6,
        Array.from(peaks).join(','));
}
{
  const frames = 480;
  const a = new Float32Array(frames).fill(0.25);
  const b = new Float32Array(frames).fill(-0.5);
  const wav = Trim.buildWav16([a, b], RATE);
  const view = new DataView(wav);
  const tag = (o, n) => String.fromCharCode(...new Uint8Array(wav, o, n));
  check('WAV 头 RIFF/WAVE',
        tag(0, 4) === 'RIFF' && tag(8, 4) === 'WAVE' && tag(12, 4) === 'fmt ' && tag(36, 4) === 'data');
  check('WAV PCM / 2 声道 / 48 kHz / 16 bit',
        view.getUint16(20, true) === 1 && view.getUint16(22, true) === 2 && view.getUint32(24, true) === RATE &&
        view.getUint16(34, true) === 16);
  check('WAV 块对齐 = 4', view.getUint16(32, true) === 4);
  check('WAV data 长度 = 帧数 × 4', view.getUint32(40, true) === frames * 4, `${view.getUint32(40, true)}`);
  check('WAV 样点正确', view.getInt16(44, true) === Math.round(0.25 * 32767) &&
                        view.getInt16(46, true) === Math.round(-0.5 * 32767));
}
check('文件名猜音色名', Trim.suggestName('DeepSeek 语音.mp3') === 'deepseek' &&
                        Trim.suggestName('录音 001.wav') === '001' &&
                        Trim.suggestName('???').length > 0);

// ---------------------------------------------------------------- 元素 id 契约
console.log('[2] index.html / app.js 的 id 契约');
{
  const html = fs.readFileSync(path.join(ROOT, 'web/app/index.html'), 'utf8');
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
  const app = fs.readFileSync(path.join(ROOT, 'web/app/app.js'), 'utf8');
  const used = new Set([...app.matchAll(/\$\('([^']+)'\)/g)].map((m) => m[1]));
  const missing = [...used].filter((id) => !ids.has(id));
  check(`app.js 用到的 ${used.size} 个 id 都存在`, missing.length === 0, missing.join(', '));
  check('index.html 在 app.js 之前加载 clone-audio.js',
        html.indexOf('clone-audio.js') > 0 && html.indexOf('clone-audio.js') < html.indexOf('app.js'));
  const script = fs.readFileSync(path.join(ROOT, 'web/app/clone-audio.js'), 'utf8');
  check('clone-audio.js 不碰 DOM', !/document\.|window\./.test(script.replace(/typeof window/g, '')));
}

// ---------------------------------------------------------------- 真引擎
async function post(pathname, body) {
  const res = await fetch(BASE + pathname, {
    method: 'POST',
    headers: { 'Content-Type': 'audio/wav' },
    body,
  });
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch { /* 错误响应可能是空的 */ }
  return { status: res.status, json, text };
}

function tone(seconds) {
  const frames = Math.round(seconds * RATE);
  const left = new Float32Array(frames);
  const right = new Float32Array(frames);
  for (let i = 0; i < frames; ++i) {
    const v = 0.25 * Math.sin((2 * Math.PI * 220 * i) / RATE);
    left[i] = v;
    right[i] = v;
  }
  return [left, right];
}

/**
 * 伪造一个 MP3-in-WAV：fmt tag 0x0055、block_align 1、bits 0。
 * 这正是 data/clone_wav 里那些样本的形态，也是当初"块对齐不符合预期"的来源。
 * 测试里现造，免得依赖 gitignored 的 data/。
 */
function mp3InWav(payload = 4096) {
  const buffer = new ArrayBuffer(44 + payload);
  const view = new DataView(buffer);
  const str = (offset, s) => {
    for (let i = 0; i < s.length; ++i) view.setUint8(offset + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  view.setUint32(4, 36 + payload, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 0x0055, true);   // MP3
  view.setUint16(22, 2, true);
  view.setUint32(24, 44100, true);
  view.setUint32(28, 40000, true);
  view.setUint16(32, 1, true);        // block_align：压缩流是 1
  view.setUint16(34, 0, true);        // bits：压缩流是 0
  str(36, 'data');
  view.setUint32(40, payload, true);
  return buffer;
}

async function voices() {
  const res = await fetch(BASE + '/api/v1/voices');
  return (await res.json()).user || [];
}

async function waitReady() {
  for (let i = 0; i < 240; ++i) {
    try {
      const res = await fetch(BASE + '/api/v1/health');
      if (res.ok) return true;
    } catch { /* 还没起来 */ }
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

console.log('[3] 真引擎（moss_tts --serve）');
if (!fs.existsSync(EXE)) throw new Error(`找不到 ${EXE}，先跑 csrc/build.bat`);
fs.mkdirSync(WORK, { recursive: true });
const server = spawn(EXE, ['--models', MODELS, '--serve', '--no-open', '--port', String(PORT),
                          '--web', path.join(ROOT, 'web/app'), '--cache', path.join(WORK, 'cache'),
                          '--data', path.join(WORK, 'userdata')], { stdio: 'ignore' });

try {
  if (!(await waitReady())) throw new Error('引擎没起来（端口被占用？）');
  const before = new Set((await voices()).map((v) => v.id));
  const name = 'sim' + Date.now().toString(36);

  // a) 12 秒整段
  {
    const [l, r] = tone(12);
    const res = await post(`/api/v1/voices/clone?name=${name}`, Trim.buildWav16([l, r], RATE));
    check('12 秒整段被接受', res.status === 200, `${res.status} ${res.text.slice(0, 120)}`);
    const found = (await voices()).find((v) => v.id === name);
    check('12 秒 → 150 帧', found && found.frames === 150, found ? `${found.frames}` : '缺失');
    if (found) before.add(name);
  }

  // b) 25 秒里选 8 秒窗口
  {
    const [l, r] = tone(25);
    const [wl, wr] = Trim.sliceWindow([l, r], { start: 12, end: 20 });
    const winName = name + 'w';
    const res = await post(`/api/v1/voices/clone?name=${winName}`, Trim.buildWav16([wl, wr], RATE));
    check('8 秒窗口被接受', res.status === 200, `${res.status} ${res.text.slice(0, 120)}`);
    const found = (await voices()).find((v) => v.id === winName);
    check('8 秒窗口 → 100 帧', found && found.frames === 100, found ? `${found.frames}` : '缺失');
  }

  // c) 25 秒整段：引擎必须拒绝（客户端只该传 20 秒以内）
  {
    const [l, r] = tone(25);
    const res = await post(`/api/v1/voices/clone?name=${name}long`, Trim.buildWav16([l, r], RATE));
    check('超过 20 秒被拒绝', res.status === 400 && /20/.test(res.text), `${res.status} ${res.text.slice(0, 160)}`);
  }

  // d) MP3-in-WAV：说清楚是编码问题，而不是"块对齐不符"
  {
    const res = await post(`/api/v1/voices/clone?name=${name}mp3`, mp3InWav());
    check('MP3-in-WAV 报编码问题', res.status === 400 && /MP3/.test(res.text),
          `${res.status} ${res.text.slice(0, 160)}`);
  }

  // e) 清理：把这次建的用户音色删掉，别污染 runs/userdata
  {
    const dir = path.join(WORK, 'userdata', 'voices', 'user');
    for (const file of fs.existsSync(dir) ? fs.readdirSync(dir) : []) {
      if (file.startsWith('sim')) fs.rmSync(path.join(dir, file), { force: true });
    }
  }
} finally {
  server.kill();
}

console.log('');
if (failures) {
  console.log(`[x] ${failures} 项失败`);
  process.exit(1);
}
console.log('[ok] 克隆链路通过');
