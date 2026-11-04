// 流式 PCM 播放器：环形缓冲 + 欠载统计。
// 主线程 push 交织立体声 Float32，process() 每 128 帧取一次。
class PcmPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capacity = 48000 * 2 * 4; // 4 秒交织立体声
    this.buf = new Float32Array(this.capacity);
    this.read = 0;
    this.write = 0;
    this.count = 0;
    this.underruns = 0;
    this.frames = 0;
    this.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'push') this.push(msg.samples);
      else if (msg.type === 'reset') { this.read = this.write = this.count = 0; this.underruns = 0; this.frames = 0; }
    };
  }

  push(samples) {
    const n = samples.length;
    for (let i = 0; i < n; ++i) {
      if (this.count >= this.capacity) break; // 满了就丢（主线程有背压，正常不会走到这里）
      this.buf[this.write] = samples[i];
      this.write = (this.write + 1) % this.capacity;
      ++this.count;
    }
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const channels = out.length;
    const frames = out[0].length;
    const need = frames * 2;
    if (this.count < need) {
      for (let c = 0; c < channels; ++c) out[c].fill(0);
      this.underruns++;
      if (this.underruns % 32 === 0) this.port.postMessage({ type: 'stats', underruns: this.underruns, frames: this.frames });
      return true;
    }
    for (let i = 0; i < frames; ++i) {
      const left = this.buf[this.read];
      this.read = (this.read + 1) % this.capacity;
      const right = this.buf[this.read];
      this.read = (this.read + 1) % this.capacity;
      this.count -= 2;
      if (channels >= 2) {
        out[0][i] = left;
        out[1][i] = right;
      } else {
        out[0][i] = 0.5 * (left + right);
      }
    }
    this.frames += frames;
    if (this.frames % (48000 * 2) < frames) {
      this.port.postMessage({ type: 'stats', underruns: this.underruns, frames: this.frames });
    }
    return true;
  }
}

registerProcessor('pcm-player', PcmPlayer);
