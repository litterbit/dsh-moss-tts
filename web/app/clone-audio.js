// 克隆参考音频的纯逻辑：裁剪窗口、波形峰值、PCM16 WAV 打包。
// 没有 DOM 依赖，浏览器里由 index.html 直接加载；Node 回归（tools/clone_sim.mjs）跑同一份文件，
// 所以这里的每个函数都必须能在没有 window / AudioContext 的环境里调用。
(function (root) {
  'use strict';

  const MAX_SECONDS = 20;   // 参考音频上限：再长只会拉长 prompt、变慢，音色不额外变好
  const MIN_SECONDS = 1;    // 再短编码出来就没帧了
  const RATE = 48000;       // 引擎统一按 48 kHz 立体声吃参考音频

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
  const num = (v, fallback) => (Number.isFinite(Number(v)) ? Number(v) : fallback);

  /** 打开文件时的默认选中：从头取到 20 s（短音频就是整段）。 */
  function defaultWindow(total) {
    const dur = Math.max(0, num(total, 0));
    return { start: 0, end: Math.min(dur, MAX_SECONDS) };
  }

  /**
   * 拖动某一端（which = 'start' | 'end'）之后的新窗口。
   * 规则说人话：
   *   - 拖起点 = 从左边裁掉（终点不动）；拖终点 = 从右边裁掉（起点不动）。
   *   - 已经顶到 20 s 上限时继续往外拖，另一端会让位，于是 20 s 窗口整段「滑」过去。
   *     长音频里选后段就是这么操作：把终点一直往右拖。
   *   - 两端不越过彼此（最短 MIN_SECONDS），也不出文件边界。
   */
  function dragWindow(total, range, which, value) {
    const dur = Math.max(0, num(total, 0));
    const limit = Math.min(MAX_SECONDS, dur);
    const minLen = Math.min(MIN_SECONDS, limit);
    const cur = range || defaultWindow(dur);
    let start = clamp(num(cur.start, 0), 0, dur);
    let end = clamp(num(cur.end, dur), 0, dur);
    if (end - start > limit) end = start + limit;

    if (which === 'start') {
      start = clamp(num(value, start), 0, Math.max(0, end - minLen));
      if (end - start > limit) end = Math.min(dur, start + limit);
    } else {
      end = clamp(num(value, end), Math.min(dur, start + minLen), dur);
      if (end - start > limit) start = Math.max(0, end - limit);
    }
    if (end < start) end = start;
    return { start, end };
  }

  /** 整段平移（拖动选中区域中间）：保持长度，只夹在文件边界内。 */
  function moveWindow(total, range, startSeconds) {
    const dur = Math.max(0, num(total, 0));
    const cur = range || defaultWindow(dur);
    const len = Math.min(Math.max(num(cur.end, 0) - num(cur.start, 0), Math.min(MIN_SECONDS, dur)), dur);
    const start = clamp(num(startSeconds, 0), 0, Math.max(0, dur - len));
    return { start, end: start + len };
  }

  /** 取出窗口对应的各声道样点（浮点，-1~1）。裁剪不复制，返回的是 subarray 视图。 */
  function sliceWindow(channels, range) {
    const rate = RATE;
    const len = channels.length ? channels[0].length : 0;
    const start = Math.round(clamp(num(range && range.start, 0), 0, len / rate) * rate);
    const end = Math.round(clamp(num(range && range.end, 0), 0, len / rate) * rate);
    return channels.map((data) => data.subarray(Math.min(start, end), Math.max(start, end)));
  }

  /** 每个像素列的最大幅值（0~1），画波形用。多个声道取最大。 */
  function peaks(channels, width) {
    const cols = Math.max(1, width | 0);
    const out = new Float32Array(cols);
    if (!channels.length || !channels[0].length) return out;
    const len = channels[0].length;
    for (let x = 0; x < cols; ++x) {
      const from = Math.floor((x * len) / cols);
      const to = Math.max(from + 1, Math.floor(((x + 1) * len) / cols));
      let peak = 0;
      for (const data of channels) {
        for (let i = from; i < to && i < len; ++i) {
          const v = data[i] < 0 ? -data[i] : data[i];
          if (v > peak) peak = v;
        }
      }
      out[x] = Math.min(1, peak);
    }
    return out;
  }

  /** 交错 PCM16 WAV（44 字节头）。引擎只吃 PCM，压缩格式在这里就变成裸样点。 */
  function buildWav16(channels, rate) {
    const ch = channels.length;
    const frames = ch ? channels[0].length : 0;
    const bytes = frames * ch * 2;
    const buffer = new ArrayBuffer(44 + bytes);
    const view = new DataView(buffer);
    const str = (offset, s) => {
      for (let i = 0; i < s.length; ++i) view.setUint8(offset + i, s.charCodeAt(i));
    };
    str(0, 'RIFF');
    view.setUint32(4, 36 + bytes, true);
    str(8, 'WAVE');
    str(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);              // PCM
    view.setUint16(22, ch, true);
    view.setUint32(24, rate || RATE, true);
    view.setUint32(28, (rate || RATE) * ch * 2, true);
    view.setUint16(32, ch * 2, true);         // block align
    view.setUint16(34, 16, true);
    str(36, 'data');
    view.setUint32(40, bytes, true);
    let offset = 44;
    for (let i = 0; i < frames; ++i) {
      for (let c = 0; c < ch; ++c) {
        const v = clamp(channels[c][i], -1, 1);
        view.setInt16(offset, Math.round(v * 32767), true);
        offset += 2;
      }
    }
    return buffer;
  }

  /** 从文件名猜一个合法音色名（引擎只收字母/数字/_/-）。 */
  function suggestName(fileName) {
    const base = String(fileName || '')
      .replace(/\.[^.]+$/, '')
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 32);
    return base || 'myvoice';
  }

  function formatSeconds(value) {
    const v = Math.max(0, num(value, 0));
    return v.toFixed(1) + ' s';
  }

  const api = {
    MAX_SECONDS,
    MIN_SECONDS,
    RATE,
    defaultWindow,
    dragWindow,
    moveWindow,
    sliceWindow,
    peaks,
    buildWav16,
    suggestName,
    formatSeconds,
  };

  root.MossCloneAudio = api;
  if (typeof module === 'object' && module && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
