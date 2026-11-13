// MOSS-TTS 本地试听台的前端逻辑。
// 与服务的契约见 doc/产品化与分发方案.md 的 API v1：
//   POST /api/v1/tts/stream  -> chunked s16le / 48000 / 2ch
//   GET  /api/v1/jobs/{id}   -> 统计
//   POST /api/v1/jobs/{id}/cancel
(() => {
  const $ = (id) => document.getElementById(id);
  // 同一个页面跑在两种地方：
  //   1) 本机服务托管的页面 → 同源，API = location.origin
  //   2) GitHub Pages（或 file://）→ 跨源连 http://127.0.0.1:8759（端口占用时会顺延）
  const LOOPBACK = ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname);
  const LOCAL_PORTS = [8759, 8760, 8761, 8762, 8763];
  const PACE_SECONDS = 1.5;   // 边播边算时允许领先播放位置多少秒
  let API = LOOPBACK ? location.origin : '';

  const state = {
    voices: { builtin: [], user: [] },
    ctx: null,
    player: null,     // StreamPlayer（player.js）：负责流量控制与进度
    controller: null,
    jobId: null,
    wav: [],          // Int16Array 分片，用于保存
    history: [],
  };

  function setStatus(text, kind = '') {
    const el = $('status');
    el.textContent = text;
    el.className = 'status' + (kind ? ' ' + kind : '');
  }

  function setEngine(text, kind = '') {
    $('engine').textContent = text;
    $('dot').className = 'dot' + (kind ? ' ' + kind : '');
  }

  function setOffline(show) {
    $('offline').classList.toggle('hidden', !show);
    if (show && LOOPBACK) {
      $('offline-hint').textContent = '本地页面已经打开了，但服务没有响应；可能是进程刚退出，重启 moss_tts.exe 再试。';
    } else if (show) {
      $('offline-hint').textContent = '这个网页只是遥控器，合成要在本机进程里跑。装好之后双击 moss_tts.exe，再点"重新检测"。';
    }
  }

  function withTimeout(ms) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ms);
    return { signal: controller.signal, done: () => clearTimeout(timer) };
  }

  /** 在本机端口上找一个活着的引擎；本机托管时直接同源。 */
  async function findLocalEngine() {
    if (LOOPBACK) return location.origin;
    const probes = LOCAL_PORTS.map((port) => {
      const base = `http://127.0.0.1:${port}`;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 900);
      return fetch(`${base}/api/v1/health`, { signal: controller.signal })
        .then((res) => (res.ok ? base : null))
        .catch(() => null)
        .finally(() => clearTimeout(timer));
    });
    const results = await Promise.all(probes);
    return results.find((value) => value !== null) || '';
  }

  async function probe() {
    API = await findLocalEngine();
    if (!API) {
      setEngine('未检测到本地引擎', 'err');
      setOffline(true);
      $('go').disabled = true;
      setStatus('本地引擎没有运行', 'err');
      return false;
    }
    const t = withTimeout(1500);
    try {
      const res = await fetch(API + '/api/v1/health', { signal: t.signal });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const info = await (await fetch(API + '/api/v1/info', { signal: t.signal })).json();
      const engine = info.engine || {};
      setEngine(`引擎 ${info.version} · ${engine.llm_threads}+${engine.decode_threads} 线程 · ${engine.backend}`,
                'ok');
      $('perf').hidden = false;
      $('perf').textContent = `${engine.physical_cores} 物理核 · 队列 ${info.queue ? info.queue.max : '?'}`;
      $('go').disabled = false;
      setOffline(false);
      return true;
    } catch (err) {
      setEngine('未检测到本地引擎', 'err');
      setOffline(true);
      $('go').disabled = true;
      setStatus('本地引擎没有运行', 'err');
      return false;
    } finally {
      t.done();
    }
  }

  function voiceValue() {
    const raw = $('voice').value || '';
    const idx = raw.indexOf(':');
    return { kind: raw.slice(0, idx), id: raw.slice(idx + 1) };
  }

  async function loadVoices() {
    const res = await fetch(API + '/api/v1/voices');
    if (!res.ok) throw new Error('音色列表读取失败');
    const data = await res.json();
    state.voices = { builtin: data.builtin || [], user: data.user || [], default: data.default || '' };
    const sel = $('voice');
    const previous = sel.value;
    sel.innerHTML = '';
    const text = (v, kind) => {
      if (kind === 'user') return `${v.id} · 克隆`;
      // 内置音色显示中文名，客户端 id 放在括号里（API 里说话要用它）
      return v.label && v.label !== v.id ? `${v.label}（${v.id}）` : v.id;
    };
    const addGroup = (label, items, kind) => {
      if (!items.length) return;
      const group = document.createElement('optgroup');
      group.label = label;
      for (const v of items) {
        const option = document.createElement('option');
        option.value = `${kind}:${v.id}`;
        option.textContent = text(v, kind);
        option.title = kind === 'user' ? `克隆音色 ${v.id}` : `内置音色 ${v.id}`;
        group.appendChild(option);
      }
      sel.appendChild(group);
    };
    addGroup(`内置音色（${state.voices.builtin.length}）`, state.voices.builtin, 'builtin');
    addGroup(`我的音色（${state.voices.user.length}）`, state.voices.user, 'user');
    // 选中顺序：上次用的 → 服务端说的默认音色（manqu/满满）→ 第一个内置 → 第一个用户音色
    const options = [...sel.options];
    const has = (value) => options.some((o) => o.value === value);
    let chosen = previous && has(previous) ? previous : '';
    if (!chosen) {
      const flagged = state.voices.builtin.find((v) => v.default) ||
                      state.voices.builtin.find((v) => v.id === (state.voices.default || 'manqu'));
      if (flagged) chosen = `builtin:${flagged.id}`;
      else if (options.length) chosen = options[0].value;
    }
    if (chosen) sel.value = chosen;
  }

  async function initAudio() {
    if (state.player) return;
    // 优先 48 kHz；个别设备/浏览器不接受指定采样率，就退回默认并在 player 里重采样
    let ctx;
    try {
      ctx = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
    } catch (err) {
      ctx = new AudioContext({ latencyHint: 'interactive' });
    }
    if (!ctx.audioWorklet) throw new Error('这个浏览器不支持 AudioWorklet');
    await ctx.audioWorklet.addModule('pcm-worklet.js');
    const node = new AudioWorkletNode(ctx, 'pcm-player', { outputChannelCount: [2] });
    node.connect(ctx.destination);
    state.ctx = ctx;
    state.player = new StreamPlayer(ctx, node);
  }

  function resetPlayer() {
    if (state.player) state.player.reset();
    state.wav = [];
    $('level').style.width = '0%';
  }

  function appendWav(int16) {
    state.wav.push(int16);
  }

  function buildWav() {
    let total = 0;
    for (const part of state.wav) total += part.length;
    const buffer = new ArrayBuffer(44 + total * 2);
    const view = new DataView(buffer);
    const writeStr = (offset, s) => { for (let i = 0; i < s.length; ++i) view.setUint8(offset + i, s.charCodeAt(i)); };
    writeStr(0, 'RIFF'); view.setUint32(4, 36 + total * 2, true); writeStr(8, 'WAVEfmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 2, true);
    view.setUint32(24, 48000, true); view.setUint32(28, 48000 * 4, true);
    view.setUint16(32, 4, true); view.setUint16(34, 16, true); writeStr(36, 'data');
    view.setUint32(40, total * 2, true);
    const out = new Int16Array(buffer, 44);
    let offset = 0;
    for (const part of state.wav) { out.set(part, offset); offset += part.length; }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  async function refreshJob() {
    if (!state.jobId) return null;
    try {
      const res = await fetch(API + '/api/v1/jobs/' + encodeURIComponent(state.jobId));
      if (!res.ok) return null;
      return await res.json();
    } catch { return null; }
  }

  function renderChips(info, extra) {
    const chips = $('chips');
    const items = [];
    if (info) {
      const audio = (info.audio_ms / 1000).toFixed(2);
      items.push(['音频', audio + ' s']);
      items.push(['帧数', info.frames]);
      items.push(['首块', info.first_pcm_ms >= 0 ? Math.round(info.first_pcm_ms) + ' ms' : '—']);
      items.push(['计算', Math.round(info.compute_ms) + ' ms']);
      items.push(['RTF', Number(info.rtf).toFixed(3)]);
      items.push(['缓存', info.cache_hit ? '命中' : '冷启动']);
      if (info.state === 'cancelled') items.push(['状态', '已停止']);
      if (info.state === 'failed') items.push(['状态', '失败']);
    }
    if (extra) items.push(...extra);
    chips.hidden = !items.length;
    chips.innerHTML = items.map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join('');
  }

  async function generate() {
    const text = $('text').value.trim();
    if (!text) { setStatus('文本为空', 'err'); return; }
    try {
      await initAudio();
    } catch (err) {
      setStatus('音频初始化失败：' + (err.message || err), 'err');
      return;
    }
    if (state.ctx.state === 'suspended') await state.ctx.resume();
    if (state.ctx.state !== 'running') {
      setStatus('浏览器没放行音频输出，先点一下页面再试', 'err');
      return;
    }
    resetPlayer();

    $('go').disabled = true;
    $('stop').disabled = false;
    $('save').disabled = true;
    $('chips').hidden = true;
    setStatus('生成中…');

    const { kind, id } = voiceValue();
    const body = {
      text,
      seed: Number($('seed').value) || 0,
      max_frames: Number($('frames').value) || 375,
      temperature: Number($('temp').value),
      top_p: Number($('topp').value),
      // 边播边算：只比播放位置快 1.5 s（服务端按墙钟节流，CPU 真的闲下来）；
      // 不勾就一次算完，速度最快但会一直占满 CPU。
      pace: $('pace').checked ? PACE_SECONDS : 0,
    };
    if (kind === 'user') { body.scope = 'user'; body.voice = id; }
    else body.voice = id;

    const controller = new AbortController();
    state.controller = controller;
    state.jobId = null;
    const t0 = performance.now();
    let firstByte = -1;
    let failure = null;

    try {
      const res = await fetch(API + '/api/v1/tts/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.error || `HTTP ${res.status}`);
      }
      state.jobId = res.headers.get('X-Moss-Request-Id');
      const reader = res.body.getReader();
      let pending = new Uint8Array(0);
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (firstByte < 0) firstByte = performance.now() - t0;
        const all = new Uint8Array(pending.length + value.length);
        all.set(pending);
        all.set(value, pending.length);
        const usable = all.length - (all.length % 4);
        if (usable > 0) {
          const slice = all.slice(0, usable);
          const int16 = new Int16Array(slice.buffer, 0, usable / 2);
          const samples = new Float32Array(usable / 2);
          let peak = 0;
          for (let i = 0; i < samples.length; ++i) {
            const v = int16[i] / 32768;
            samples[i] = v;
            const a = v < 0 ? -v : v;
            if (a > peak) peak = a;
          }
          await state.player.send(samples);
          appendWav(new Int16Array(int16));
          $('level').style.width = Math.min(100, peak * 140) + '%';
        }
        pending = all.slice(usable);
      }
      state.player.end();   // 让最后不足一块的尾巴也能播出来
    } catch (err) {
      if (err.name !== 'AbortError') failure = err.message || String(err);
    } finally {
      state.controller = null;
      $('stop').disabled = true;
      $('go').disabled = false;
      $('level').style.width = '0%';
      $('save').disabled = state.wav.length === 0;
    }

    if (failure) {
      setStatus('失败：' + failure, 'err');
      return;
    }

    const info = await refreshJob();
    if (!info) { setStatus('完成（统计不可用）'); return; }
    if (info.state === 'failed') { setStatus('失败：' + (info.error || '推理失败'), 'err'); return; }
    // 墙钟明显大于纯计算时间 = 服务端在按播放进度节流（边播边算）
    const wall = Number(info.wall_ms) || Number(info.compute_ms);
    const paced = wall > info.compute_ms + 400;
    if (info.state === 'cancelled') setStatus('已停止', 'ok');
    else if (paced) {
      setStatus(`完成：${(info.audio_ms / 1000).toFixed(2)} s，计算 ${Math.round(info.compute_ms)} ms（边播边算，墙钟 ${(wall / 1000).toFixed(2)} s）`, 'ok');
    } else {
      setStatus(`完成：${(info.audio_ms / 1000).toFixed(2)} s，计算 ${Math.round(info.compute_ms)} ms，RTF ${Number(info.rtf).toFixed(3)}`, 'ok');
    }
    const extra = [];
    if (firstByte >= 0) extra.push(['首包到达', Math.round(firstByte) + ' ms']);
    if (paced) extra.push(['墙钟', (wall / 1000).toFixed(2) + ' s']);
    if (state.player && state.player.dropped > 0) extra.push(['丢样', state.player.dropped + ' 帧']);
    if (state.player && state.player.underruns > 0) extra.push(['欠载', state.player.underruns + ' 次']);
    renderChips(info, extra);
    if (info.state === 'completed') {
      pushHistory({ text, voice: $('voice').selectedOptions[0]?.textContent || '', info });
    }
  }

  async function stop() {
    if (state.controller) state.controller.abort();
    if (state.jobId) {
      try {
        await fetch(API + '/api/v1/jobs/' + encodeURIComponent(state.jobId) + '/cancel', { method: 'POST' });
      } catch { /* 忽略 */ }
    }
    resetPlayer();
    $('stop').disabled = true;
    setStatus('已停止', 'ok');
  }

  function pushHistory(entry) {
    state.history.unshift(entry);
    state.history = state.history.slice(0, 8);
    const hist = $('hist');
    hist.innerHTML = '';
    for (const item of state.history) {
      const row = document.createElement('div');
      row.className = 'item';
      const txt = document.createElement('span');
      txt.className = 'txt';
      txt.textContent = `${item.voice} · ${item.text}`;
      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = `${(item.info.audio_ms / 1000).toFixed(2)} s · RTF ${Number(item.info.rtf).toFixed(3)}`;
      row.appendChild(txt);
      row.appendChild(meta);
      row.onclick = () => { $('text').value = item.text; };
      hist.appendChild(row);
    }
  }

  // ---- 音色克隆：拖入音频 → 浏览器解码 → 在波形上选一段（≤20 s）→ 本地编码 ----------
  // 为什么在浏览器里解码：引擎是零依赖的 C++，装不下 MP3/AAC 解码器；而浏览器的
  // decodeAudioData 什么格式都能解，还能顺手把采样率转成 48 kHz。引擎只收 PCM WAV。
  const Trim = window.MossCloneAudio;
  const clone = {
    channels: [],            // 48 kHz 浮点样点（AudioBuffer 的视图，不复制）
    total: 0,                // 秒
    window: { start: 0, end: 0 },
    peaks: null,
    peakCols: 0,
    ctx: null,               // 试听用的 AudioContext（懒建）
    preview: null,
    raf: 0,
    drag: null,              // 'start' | 'end' | 'move'
    anchor: 0,
    busy: false,
  };

  function cloneStatus(text, kind = '') {
    const el = $('clone-status');
    el.textContent = text;
    el.className = 'status' + (kind ? ' ' + kind : '');
  }

  function cloneTip(text) { $('clone-tip').textContent = text; }

  function stopPreview() {
    if (clone.raf) { cancelAnimationFrame(clone.raf); clone.raf = 0; }
    if (clone.preview) {
      try { clone.preview.source.stop(); } catch { /* 已经停了 */ }
      try { clone.preview.source.disconnect(); } catch { /* 忽略 */ }
      clone.preview = null;
    }
    $('clone-play').textContent = '试听这一段';
    if (clone.channels.length) drawWave();
  }

  /** 波形：高亮选中区、两端手柄、试听播放头。全程设备像素，1 px 柱状。 */
  function drawWave(playhead) {
    const canvas = $('clone-canvas');
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round((canvas.clientWidth || 640) * dpr));
    const height = Math.max(1, Math.round((canvas.clientHeight || 88) * dpr));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    if (!clone.channels.length || !clone.total) return;
    if (!clone.peaks || clone.peakCols !== width) {
      clone.peaks = Trim.peaks(clone.channels, width);
      clone.peakCols = width;
    }
    const startX = (clone.window.start / clone.total) * width;
    const endX = (clone.window.end / clone.total) * width;
    ctx.fillStyle = 'rgba(79,140,255,0.15)';
    ctx.fillRect(startX, 0, Math.max(1, endX - startX), height);
    const mid = height / 2;
    for (let x = 0; x < width; ++x) {
      const inside = x >= startX && x <= endX;
      const amp = Math.max(0.04, clone.peaks[x]);
      const bar = Math.max(1, amp * (height - 6));
      ctx.fillStyle = inside ? '#7ba7ff' : '#39415a';
      ctx.fillRect(x, mid - bar / 2, 1, bar);
    }
    ctx.fillStyle = '#4f8cff';
    ctx.fillRect(Math.max(0, Math.round(startX) - 1), 0, 2, height);
    ctx.fillRect(Math.min(width - 2, Math.round(endX) - 1), 0, 2, height);
    if (playhead != null && playhead >= 0 && playhead <= clone.total) {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(Math.round((playhead / clone.total) * width), 0, 1, height);
    }
  }

  function syncTrim() {
    const { start, end } = clone.window;
    const total = clone.total;
    $('clone-range').textContent = `${start.toFixed(1)} – ${end.toFixed(1)} s`;
    $('clone-len').textContent = Trim.formatSeconds(end - start);
    $('clone-start-v').textContent = Trim.formatSeconds(start);
    $('clone-end-v').textContent = Trim.formatSeconds(end);
    for (const [id, value] of [['clone-start', start], ['clone-end', end]]) {
      const el = $(id);
      el.max = total.toFixed(2);
      el.value = value.toFixed(2);
    }
    $('clone-go').disabled = clone.busy || !clone.channels.length || end - start < 0.5;
    drawWave();
  }

  function setWindow(next) {
    clone.window = next;
    stopPreview();
    syncTrim();
  }

  function timeAt(clientX) {
    const rect = $('clone-canvas').getBoundingClientRect();
    const ratio = rect.width ? (clientX - rect.left) / rect.width : 0;
    return Math.max(0, Math.min(clone.total, ratio * clone.total));
  }

  function cloneReset() {
    stopPreview();
    clone.channels = [];
    clone.total = 0;
    clone.window = { start: 0, end: 0 };
    clone.peaks = null;
    clone.peakCols = 0;
    $('clone-editor').classList.add('hidden');
    $('clone-range').textContent = '0.0 – 0.0 s';
    $('clone-len').textContent = '0.0 s';
    $('clone-file-name').textContent = '';
    $('clone-go').disabled = true;
    cloneTip('');
  }

  /** OfflineAudioContext 自带采样率换算：MP3/M4A/AAC/OGG/FLAC 解出来就是 48 kHz。 */
  async function decodeClip(file) {
    if (file.size > 100 * 1024 * 1024) throw new Error('文件超过 100 MB，先截一段再拖进来');
    const bytes = await file.arrayBuffer();
    const off = new OfflineAudioContext(2, Trim.RATE, Trim.RATE);
    const buffer = await off.decodeAudioData(bytes);
    if (buffer.duration > 900) {
      throw new Error(`这段音频 ${buffer.duration.toFixed(0)} 秒太长了，先用剪辑软件截一段`);
    }
    return buffer;
  }

  async function loadClip(file) {
    if (!file) return;
    cloneStatus('正在解码音频…');
    try {
      const buffer = await decodeClip(file);
      stopPreview();
      const left = buffer.getChannelData(0);
      const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
      clone.channels = [left, right];
      clone.total = Math.max(0.001, left.length / Trim.RATE);
      clone.peaks = null;
      clone.peakCols = 0;
      clone.window = Trim.defaultWindow(clone.total);
      $('clone-file-name').textContent = `${file.name} · ${clone.total.toFixed(1)} s`;
      $('clone-editor').classList.remove('hidden');
      if (!($('clone-name').value || '').trim()) $('clone-name').value = Trim.suggestName(file.name);
      syncTrim();
      const long = clone.total > Trim.MAX_SECONDS;
      cloneTip(long ? '超过 20 秒：把「终点」往右拖，20 秒窗口会整段滑过去' : '整段都能用，也可以再剪短一点');
      cloneStatus(`已载入 ${clone.total.toFixed(1)} s${long ? '，请选一段（≤20 s）' : ''}`, 'ok');
    } catch (err) {
      cloneReset();
      cloneStatus('读不了这个文件：' + (err && err.message ? err.message : err), 'err');
      if (!String(err).includes('太长') && !String(err).includes('MB')) {
        cloneTip('浏览器不支持这个编码时可以先用工具转成 MP3 或 WAV');
      }
    }
  }

  async function playPreview() {
    if (!clone.channels.length) return;
    stopPreview();
    try {
      if (!clone.ctx) {
        try { clone.ctx = new AudioContext({ sampleRate: Trim.RATE }); }
        catch { clone.ctx = new AudioContext(); }
      }
      const ctx = clone.ctx;
      if (ctx.state === 'suspended') await ctx.resume();
      const [left, right] = Trim.sliceWindow(clone.channels, clone.window);
      if (!left.length) return;
      const buffer = ctx.createBuffer(2, left.length, Trim.RATE);
      buffer.copyToChannel(left, 0);
      buffer.copyToChannel(right, 1);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const offset = clone.window.start;
      source.start();
      clone.preview = { source, startedAt: ctx.currentTime, length: left.length / Trim.RATE };
      $('clone-play').textContent = '停止试听';
      const tick = () => {
        if (!clone.preview) return;
        const elapsed = ctx.currentTime - clone.preview.startedAt;
        drawWave(offset + elapsed);
        if (elapsed < clone.preview.length) clone.raf = requestAnimationFrame(tick);
        else stopPreview();
      };
      tick();
    } catch (err) {
      cloneStatus('试听失败：' + (err.message || err), 'err');
    }
  }

  async function createVoice() {
    if (!clone.channels.length) { cloneStatus('先拖一段音频进来', 'err'); return; }
    const name = ($('clone-name').value || '').trim();
    if (!/^[A-Za-z0-9_-]{1,32}$/.test(name)) {
      cloneStatus('音色名只能是字母、数字、_ 或 -', 'err');
      return;
    }
    const length = clone.window.end - clone.window.start;
    if (length < 0.5) { cloneStatus('选的片段太短，至少 1 秒', 'err'); return; }
    stopPreview();
    clone.busy = true;
    $('clone-go').disabled = true;
    cloneStatus(`正在编码音色…（${length.toFixed(1)} s 参考音频）`);
    try {
      const [left, right] = Trim.sliceWindow(clone.channels, clone.window);
      const wav = Trim.buildWav16([left, right], Trim.RATE);
      const res = await fetch(`${API}/api/v1/voices/clone?name=${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: wav,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      await loadVoices();
      $('voice').value = `user:${name}`;
      const seconds = data.user?.find((v) => v.id === name)?.seconds;
      cloneStatus(`音色 ${name} 已创建${seconds ? `（${seconds.toFixed(2)} s）` : ''}`, 'ok');
      cloneTip('已存进本机用户音色目录，现在就能在上面选它');
    } catch (err) {
      cloneStatus('克隆失败：' + (err.message || err), 'err');
    } finally {
      clone.busy = false;
      $('clone-go').disabled = !clone.channels.length;
    }
  }

  // ---- 性能面板：两套线程数写进 <data>/settings.json，重启进程生效 ----------
  async function loadThreads() {
    if (!API) return;
    try {
      const [settings, info] = await Promise.all([
        fetch(API + '/api/v1/settings').then((r) => (r.ok ? r.json() : null)),
        fetch(API + '/api/v1/info').then((r) => r.json()),
      ]);
      if (!settings) throw new Error('这个引擎版本没有 /settings');
      const eng = info.engine || {};
      const llm = settings.llm_threads || eng.llm_threads || 0;
      const dec = settings.decode_threads || eng.decode_threads || 0;
      // 输入框里放"正在生效"的数字（自动模式下也是具体值），不显示 0+0
      $('th-llm').value = llm;
      $('th-decode').value = dec;
      $('th-llm-v').textContent = settings.auto ? `${llm}（自动）` : String(llm);
      $('th-decode-v').textContent = settings.auto ? `${dec}（自动）` : String(dec);
      $('th-status').className = 'status';
      $('th-status').textContent = `正在使用 ${llm}+${dec} 线程`;
    } catch {
      $('th-status').textContent = '这个引擎版本不支持在线改线程数';
    }
  }

  async function saveThreads() {
    const body = {
      llm_threads: Number($('th-llm').value) || 0,
      decode_threads: Number($('th-decode').value) || 0,
    };
    $('th-status').className = 'status';
    $('th-status').textContent = '保存中…';
    try {
      const res = await fetch(API + '/api/v1/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      const llm = data.llm_threads || 0;
      const dec = data.decode_threads || 0;
      $('th-llm-v').textContent = llm ? String(llm) : '自动';
      $('th-decode-v').textContent = dec ? String(dec) : '自动';
      $('th-status').className = 'status ok';
      $('th-status').textContent = `已保存 ${llm || '自动'}+${dec || '自动'}，点「重启引擎」生效`;
    } catch (err) {
      $('th-status').className = 'status err';
      $('th-status').textContent = '保存失败：' + (err.message || err);
    }
  }

  /** 让引擎自己重起一个进程（服务端 spawn 新进程后退出），页面跟着刷新。 */
  async function restartEngine() {
    const button = $('th-restart');
    button.disabled = true;
    $('th-status').className = 'status';
    $('th-status').textContent = '正在重启…';
    try {
      const res = await fetch(API + '/api/v1/restart', { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      button.disabled = false;
      $('th-status').className = 'status err';
      $('th-status').textContent = '重启失败：' + (err.message || err);
      return;
    }
    // 旧进程退出、新进程接管端口；端口若顺延了，页面跟着跳过去
    const deadline = Date.now() + 25000;
    const poll = async () => {
      const base = await findLocalEngine();
      if (base) {
        if (base !== location.origin) location.replace(base + '/');
        else location.reload();
        return;
      }
      if (Date.now() > deadline) {
        button.disabled = false;
        $('th-status').className = 'status err';
        $('th-status').textContent = '没等到新引擎起来，请手动重新运行 moss_tts.exe';
        return;
      }
      setTimeout(poll, 400);
    };
    setTimeout(poll, 600);
  }

  function saveWav() {
    if (!state.wav.length) return;
    const blob = buildWav();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'moss-tts.wav';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  $('go').onclick = generate;
  $('stop').onclick = stop;
  $('save').onclick = saveWav;
  $('clone-go').onclick = createVoice;
  $('th-save').onclick = saveThreads;
  $('th-restart').onclick = restartEngine;
  $('clone-play').onclick = () => (clone.preview ? stopPreview() : playPreview());
  $('clone-reset').onclick = () => {
    if (!clone.total) return;
    clone.window = { start: 0, end: Math.min(Trim.MAX_SECONDS, clone.total) };
    stopPreview();
    syncTrim();
    cloneTip(clone.total > Trim.MAX_SECONDS ? '已回到开头 20 秒；把「终点」往右拖可以把窗口往后滑'
                                            : '已选中整段');
  };
  $('clone-start').oninput = (event) =>
    setWindow(Trim.dragWindow(clone.total, clone.window, 'start', Number(event.target.value)));
  $('clone-end').oninput = (event) =>
    setWindow(Trim.dragWindow(clone.total, clone.window, 'end', Number(event.target.value)));

  // 波形上直接拖：两端是裁剪手柄，中间整段平移，外面点一下就把窗口挪过去
  const canvas = $('clone-canvas');
  const grabSeconds = () => Math.max(0.2, clone.total * 0.012);
  canvas.addEventListener('pointerdown', (event) => {
    if (!clone.channels.length) return;
    const t = timeAt(event.clientX);
    if (Math.abs(t - clone.window.start) <= grabSeconds()) clone.drag = 'start';
    else if (Math.abs(t - clone.window.end) <= grabSeconds()) clone.drag = 'end';
    else if (t > clone.window.start && t < clone.window.end) {
      clone.drag = 'move';
      clone.anchor = t - clone.window.start;
    } else {
      clone.drag = 'move';
      clone.anchor = (clone.window.end - clone.window.start) / 2;
      setWindow(Trim.moveWindow(clone.total, clone.window, t - clone.anchor));
    }
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!clone.channels.length) return;
    const t = timeAt(event.clientX);
    if (!clone.drag) {
      canvas.style.cursor = Math.abs(t - clone.window.start) <= grabSeconds() ||
                            Math.abs(t - clone.window.end) <= grabSeconds() ? 'ew-resize' : 'crosshair';
      return;
    }
    if (clone.drag === 'move') setWindow(Trim.moveWindow(clone.total, clone.window, t - clone.anchor));
    else setWindow(Trim.dragWindow(clone.total, clone.window, clone.drag, t));
  });
  const endDrag = () => { clone.drag = null; };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  const drop = $('clone-drop');
  const fileInput = $('clone-file');
  let dragDepth = 0;
  drop.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files[0];
    fileInput.value = '';   // 允许连续拖同一个文件
    loadClip(file);
  });
  drop.addEventListener('dragenter', (event) => {
    event.preventDefault();
    if (++dragDepth === 1) drop.classList.add('over');
  });
  drop.addEventListener('dragover', (event) => event.preventDefault());
  drop.addEventListener('dragleave', (event) => {
    event.preventDefault();
    if (--dragDepth <= 0) { dragDepth = 0; drop.classList.remove('over'); }
  });
  drop.addEventListener('drop', (event) => {
    event.preventDefault();
    dragDepth = 0;
    drop.classList.remove('over');
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) loadClip(file);
  });
  // 拖偏到页面空白处时，别让浏览器直接打开文件
  window.addEventListener('dragover', (event) => event.preventDefault());
  window.addEventListener('drop', (event) => event.preventDefault());
  window.addEventListener('resize', () => {
    clone.peakCols = 0;
    if (clone.channels.length) drawWave();
  });
  $('retry').onclick = async () => {
    setStatus('正在检测…');
    if (await probe()) {
      try {
        await loadVoices();
        setStatus('已连接本地引擎', 'ok');
        loadThreads();
      } catch (err) {
        setStatus('音色列表加载失败：' + (err.message || err), 'err');
      }
    }
  };
  $('temp').oninput = () => { $('temp-v').textContent = Number($('temp').value).toFixed(2); };
  $('topp').oninput = () => { $('topp-v').textContent = Number($('topp').value).toFixed(2); };
  $('text').addEventListener('keydown', (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') generate();
  });

  (async () => {
    const ready = await probe();
    if (!ready) return;
    try {
      await loadVoices();
    } catch (err) {
      setStatus('音色列表加载失败：' + (err.message || err), 'err');
    }
    loadThreads();
  })();
})();
