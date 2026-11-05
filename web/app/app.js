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
    state.voices = { builtin: data.builtin || [], user: data.user || [] };
    const sel = $('voice');
    const previous = sel.value;
    sel.innerHTML = '';
    const addGroup = (label, items, kind) => {
      if (!items.length) return;
      const group = document.createElement('optgroup');
      group.label = label;
      for (const v of items) {
        const option = document.createElement('option');
        option.value = `${kind}:${v.id}`;
        option.textContent = kind === 'user' ? `${v.id} · 克隆` : v.id;
        group.appendChild(option);
      }
      sel.appendChild(group);
    };
    addGroup(`内置音色（${state.voices.builtin.length}）`, state.voices.builtin, 'builtin');
    addGroup(`我的音色（${state.voices.user.length}）`, state.voices.user, 'user');
    if (previous && [...sel.options].some((o) => o.value === previous)) sel.value = previous;
    else if (state.voices.user.length) sel.value = `user:${state.voices.user[0].id}`;
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

  async function createVoice() {
    const file = $('clone-file').files[0];
    const name = ($('clone-name').value || '').trim();
    if (!file) { setStatus('先选一个 WAV 文件', 'err'); return; }
    if (!/^[A-Za-z0-9_-]{1,32}$/.test(name)) { setStatus('音色名只能是字母、数字、_ 或 -', 'err'); return; }
    setStatus('正在编码音色…');
    try {
      const bytes = await file.arrayBuffer();
      const res = await fetch(`${API}/api/v1/voices/clone?name=${encodeURIComponent(name)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: bytes,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      await loadVoices();
      $('voice').value = `user:${name}`;
      setStatus(`音色 ${name} 已创建（${data.user?.find((v) => v.id === name)?.seconds?.toFixed(2) || '?'} s 参考）`, 'ok');
      $('clone-name').value = '';
      $('clone-file').value = '';
    } catch (err) {
      setStatus('克隆失败：' + (err.message || err), 'err');
    }
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
  $('retry').onclick = async () => {
    setStatus('正在检测…');
    if (await probe()) {
      try {
        await loadVoices();
        setStatus('已连接本地引擎', 'ok');
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
  })();
})();
