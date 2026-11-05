// 流式 PCM 播放器（音频线程侧）：环形缓冲 + 进度回报。
//
// 只做两件事：把主线程推来的样点写进环形缓冲；每次 process() 取 128 帧输出。
// 流量控制在主线程（player.js）做信用窗口，所以正常情况这里永不溢出；
// dropped 只有主线程违约才会增长，用来抓回归。
class PcmPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capacity = 48000 * 2 * 4;   // 4 s 交织立体声
    this.buf = new Float32Array(this.capacity);
    this.read = 0;                   // 读位置（样点）
    this.write = 0;                  // 写位置（样点）
    this.count = 0;                  // 缓冲里的样点数（1 帧 = 2 个）
    this.played = 0;                 // 已输出帧数
    this.underruns = 0;              // 中途断粮次数（首帧到达前不计数）
    this.dropped = 0;                // 溢出丢掉的样点数，正常恒为 0
    this.started = false;
    this.ended = false;
    this.reported = 0;
    this.reportEvery = 2048;         // 约 43 ms 回报一次，够主线程算信用
    this.port.onmessage = (event) => this.onMessage(event.data);
  }

  onMessage(msg) {
    if (msg.type === 'push') {
      this.started = true;
      this.push(msg.samples);
    } else if (msg.type === 'end') {
      this.ended = true;
    } else if (msg.type === 'reset') {
      this.read = this.write = this.count = 0;
      this.played = this.underruns = this.dropped = 0;
      this.started = this.ended = false;
      this.report();
    }
  }

  push(samples) {
    const n = samples.length;
    for (let i = 0; i < n; ++i) {
      if (this.count >= this.capacity) {   // 主线程没守信用窗口才会到这里
        this.dropped += n - i;
        return;
      }
      this.buf[this.write] = samples[i];
      this.write = (this.write + 1) % this.capacity;
      ++this.count;
    }
  }

  report() {
    this.reported = this.played;
    this.port.postMessage({
      type: 'progress',
      played: this.played,
      underruns: this.underruns,
      dropped: Math.floor(this.dropped / 2),
      buffered: Math.floor(this.count / 2),
    });
  }

  emit(out, channels, index, left, right) {
    if (channels >= 2) {
      out[0][index] = left;
      out[1][index] = right;
    } else {
      out[0][index] = 0.5 * (left + right);
    }
  }

  process(inputs, outputs) {
    const out = outputs[0];
    const frames = out[0].length;
    const channels = out.length;
    const need = frames * 2;

    if (this.count < need) {
      // 不够一整块：流结束时把尾巴播完，否则就是欠载（输出静音）
      const tail = this.ended ? Math.floor(this.count / 2) : 0;
      for (let c = 0; c < channels; ++c) out[c].fill(0);
      for (let i = 0; i < tail; ++i) {
        const left = this.buf[this.read];
        this.read = (this.read + 1) % this.capacity;
        const right = this.buf[this.read];
        this.read = (this.read + 1) % this.capacity;
        this.count -= 2;
        this.emit(out, channels, i, left, right);
      }
      this.played += tail;
      if (tail === 0 && this.started && !this.ended) this.underruns++;
      if (this.played !== this.reported) this.report();
      return true;
    }

    for (let i = 0; i < frames; ++i) {
      const left = this.buf[this.read];
      this.read = (this.read + 1) % this.capacity;
      const right = this.buf[this.read];
      this.read = (this.read + 1) % this.capacity;
      this.count -= 2;
      this.emit(out, channels, i, left, right);
    }
    this.played += frames;
    if (this.played - this.reported >= this.reportEvery) this.report();
    return true;
  }
}

registerProcessor('pcm-player', PcmPlayer);
