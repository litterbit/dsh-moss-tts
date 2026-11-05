// 主线程侧的播放器：把「网络块」安全地送进音频线程的环形缓冲。
//
// 为什么必须单独一层：推理比实时快 6~7 倍，一段 5 s 的音频 0.8 s 就算完了。
// 如果来一块就 postMessage 一块，音频线程 4 s 的环形缓冲会被灌满，溢出的只能丢
// （pcm-worklet.js 里的 dropped）——这就是 v0.1.0「播到一半跳过一截、还没播完就结束」。
//
// 这里做信用控制：最多领先播放位置 LEAD_FRAMES，满了 send() 就等，主线程也就不再读
// socket，TCP 背压会把服务端的生成也停下来。所以播放永远是连续的，且不会丢样本。
(function (global) {
  'use strict';

  const RATE = 48000;                            // 服务端固定 s16le / 48k / 2ch
  const LEAD_FRAMES = Math.round(RATE * 2.5);    // 领先播放位置 2.5 s（环缓 4 s，留余量）

  class StreamPlayer {
    constructor(ctx, node) {
      this.ctx = ctx;
      this.node = node;
      this.pushed = 0;      // 已交给音频线程的帧数
      this.played = 0;      // 音频线程已输出的帧数
      this.underruns = 0;   // 中途断粮次数
      this.dropped = 0;     // 音频线程溢出丢的帧数，正常恒为 0
      this.waiters = [];
      this.token = 0;       // reset 后作废旧 send()
      node.port.onmessage = (event) => this._onMessage(event.data);
    }

    _onMessage(msg) {
      if (msg.type !== 'progress') return;
      this.played = msg.played;
      this.underruns = msg.underruns;
      this.dropped = msg.dropped;
      this._release();
    }

    _release() {
      if (!this.waiters.length || !this.hasRoom()) return;
      const waiters = this.waiters;
      this.waiters = [];
      for (const resolve of waiters) resolve();
    }

    backlog() { return Math.max(0, this.pushed - this.played); }
    hasRoom() { return this.backlog() < LEAD_FRAMES; }

    /** 推一段 48 kHz 交织立体声（Float32）；缓冲满时会等到播掉一些再返回。 */
    async send(samples) {
      const token = this.token;
      const out = this._resample(samples);
      while (!this.hasRoom()) await new Promise((resolve) => this.waiters.push(resolve));
      if (token !== this.token) return;   // reset 之后才解阻的旧数据，丢掉
      this.pushed += out.length / 2;
      this.node.port.postMessage({ type: 'push', samples: out });
    }

    /** 流结束：让音频线程把最后不足 128 帧的尾巴也播出来。 */
    end() { this.node.port.postMessage({ type: 'end' }); }

    reset() {
      this.token++;   // 让还在等 room 的旧 send() 醒来后直接放弃
      this.node.port.postMessage({ type: 'reset' });
      this.pushed = 0;
      this.played = 0;
      this.underruns = 0;
      this.dropped = 0;
      this._release();
    }

    // 设备只给 44.1k 时线性重采样，保证音高不变
    _resample(samples) {
      const rate = this.ctx.sampleRate;
      if (rate === RATE || samples.length < 4) return samples;
      const ratio = RATE / rate;
      const srcFrames = samples.length / 2;
      const frames = Math.floor(srcFrames / ratio);
      const out = new Float32Array(frames * 2);
      for (let i = 0; i < frames; ++i) {
        const src = i * ratio;
        const i0 = Math.floor(src);
        const i1 = Math.min(i0 + 1, srcFrames - 1);
        const frac = src - i0;
        for (let c = 0; c < 2; ++c) {
          out[i * 2 + c] = samples[i0 * 2 + c] * (1 - frac) + samples[i1 * 2 + c] * frac;
        }
      }
      return out;
    }
  }

  global.StreamPlayer = StreamPlayer;
  global.MOSS_AUDIO_RATE = RATE;
})(globalThis);
