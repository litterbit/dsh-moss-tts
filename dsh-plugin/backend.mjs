// MOSS-TTS 本地引擎的 Node 客户端（走 --stdio JSON 行协议）。
//
// 设计要点：
//   - 常驻子进程：只 spawn 一次，权重只加载一次；所有句子共用同一个进程与前缀 KV 缓存。
//   - 不占端口、不需要先起服务：父进程退出时子进程跟着走（dispose）。
//   - 流式：每个解码块回调一次 PCM（s16le / 48 kHz / 立体声），插件可以边收边播或先缓冲。
//
// 用法：
//   const tts = new TtsBackend({ exe, models })
//   await tts.start()
//   const job = tts.speak({ text: '你好', voice: 'Junhao', onAudio: (buf) => ... })
//   await job.done
//   await tts.dispose()
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export const AUDIO_FORMAT = { encoding: 's16le', rate: 48000, channels: 2 }
export const PCM_BYTES_PER_MODEL_FRAME = 3840 * 2 * 2 // 3840 采样/帧 × 2 声道 × 2 字节

function firstExisting(candidates) {
  for (const candidate of candidates) {
    if (!candidate) continue
    try {
      if (fs.existsSync(candidate)) return candidate
    } catch {}
  }
  return undefined
}

export function defaultExe() {
  const local = process.env.LOCALAPPDATA
  return firstExisting([
    process.env.MOSS_TTS_EXE,
    local ? path.join(local, 'MossTTS', 'moss_tts.exe') : undefined,
    path.resolve('moss_tts.exe'),
    path.resolve('dist', 'moss-tts-cpu-win64', 'moss_tts.exe'),
  ])
}

export function resolveModels(exe) {
  const local = process.env.LOCALAPPDATA
  return firstExisting([
    process.env.MOSS_TTS_MODELS,
    exe ? path.join(path.dirname(exe), 'models') : undefined,
    local ? path.join(local, 'MossTTS', 'models') : undefined,
    path.resolve('models'),
    path.resolve('weights'),
  ])
}

export class TtsBackend {
  constructor(options = {}) {
    this.exe = options.exe ?? defaultExe()
    this.models = options.models ?? (this.exe ? resolveModels(this.exe) : undefined)
    this.data = options.data
    this.llmThreads = options.llmThreads ?? 0
    this.decodeThreads = options.decodeThreads ?? 0
    this.logger = options.logger ?? (() => {})
    this.child = undefined
    this.pending = new Map()
    this.seq = 0
    this.buffer = ''
    this.info = undefined
    this.disposed = false
  }

  get ready() {
    return this.child !== undefined
  }

  async start() {
    if (this.child) return this.info
    if (!this.exe) throw new Error('找不到 moss_tts.exe（用 config.exe 或 MOSS_TTS_EXE 指定）')
    if (!this.models) throw new Error('找不到 models 目录（用 config.models 或 MOSS_TTS_MODELS 指定）')

    const args = ['--models', this.models, '--stdio']
    if (this.data) args.push('--data', this.data)
    if (this.llmThreads > 0) args.push('--threads', String(this.llmThreads))
    if (this.decodeThreads > 0) args.push('--decode-threads', String(this.decodeThreads))

    this.child = spawn(this.exe, args, { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
    this.child.stdout.setEncoding('utf8')
    this.child.stdout.on('data', (chunk) => this.#onData(chunk))
    this.child.stderr.setEncoding('utf8')
    this.child.stderr.on('data', (chunk) => {
      const text = String(chunk).trim()
      if (text) this.logger(`[moss-tts] ${text}`)
    })
    this.child.on('exit', (code) => {
      const error = new Error(`moss_tts 退出（code=${code}）`)
      for (const entry of this.pending.values()) entry.reject(error)
      this.pending.clear()
      this.child = undefined
    })
    this.child.on('error', (error) => {
      this.logger(`[moss-tts] spawn 失败：${error.message}`)
      this.child = undefined
    })

    this.info = await this.#request({ cmd: 'info' })
    return this.info
  }

  #onData(chunk) {
    this.buffer += chunk
    let index
    while ((index = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, index).trim()
      this.buffer = this.buffer.slice(index + 1)
      if (!line) continue
      let message
      try {
        message = JSON.parse(line)
      } catch {
        this.logger(`[moss-tts] 无法解析的行：${line.slice(0, 120)}`)
        continue
      }
      this.#dispatch(message)
    }
  }

  #dispatch(message) {
    const entry = this.pending.get(message.id)
    if (!entry) return
    if (message.type === 'audio') {
      const pcm = Buffer.from(message.pcm, 'base64')
      try {
        entry.onAudio?.(pcm)
      } catch (error) {
        this.logger(`[moss-tts] onAudio 抛错：${error?.message ?? error}`)
      }
      return
    }
    if (message.type === 'begin') {
      entry.onBegin?.(message)
      return
    }
    this.pending.delete(message.id)
    if (message.type === 'error') {
      entry.reject(new Error(`${message.code}: ${message.error}`))
    } else {
      entry.resolve(message)
    }
  }

  #write(payload) {
    if (!this.child) throw new Error('moss_tts 还没启动')
    this.child.stdin.write(`${JSON.stringify(payload)}\n`)
  }

  #request(payload, { timeoutMs = 120000 } = {}) {
    const id = payload.id ?? `r${++this.seq}`
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`moss_tts 请求超时：${payload.cmd}`))
      }, timeoutMs)
      this.pending.set(id, {
        resolve: (value) => {
          clearTimeout(timer)
          resolve(value)
        },
        reject: (error) => {
          clearTimeout(timer)
          reject(error)
        },
      })
      try {
        this.#write({ ...payload, id })
      } catch (error) {
        clearTimeout(timer)
        this.pending.delete(id)
        reject(error)
      }
    })
  }

  async voices() {
    const message = await this.#request({ cmd: 'voices' })
    return { builtin: message.builtin ?? [], user: message.user ?? [] }
  }

  async clone(name, wav) {
    const bytes = Buffer.isBuffer(wav) ? wav : fs.readFileSync(String(wav))
    return this.#request({ cmd: 'clone', name, wav: bytes.toString('base64') }, { timeoutMs: 300000 })
  }

  /**
   * 合成一句话。返回 { id, done: Promise<job>, cancel() }。
   * onAudio 在每个解码块到达时同步调用（Buffer，s16le）。
   */
  speak(options) {
    const id = `s${++this.seq}`
    const payload = {
      id,
      cmd: 'speak',
      text: options.text,
      voice: options.voice ?? 'Junhao',
      scope: options.scope ?? 'builtin',
      seed: options.seed ?? 0,
      greedy: options.greedy ?? false,
      max_frames: options.maxFrames ?? 375,
      chunk_frames: options.chunkFrames ?? 8,
      min_frames: options.minFrames ?? 0,
    }
    if (options.temperature) payload.temperature = options.temperature
    if (options.topK) payload.top_k = options.topK
    if (options.topP) payload.top_p = options.topP
    if (options.repetitionPenalty) payload.repetition_penalty = options.repetitionPenalty

    const done = new Promise((resolve, reject) => {
      this.pending.set(id, {
        onAudio: options.onAudio,
        onBegin: options.onBegin,
        resolve: (message) => resolve(message.job),
        reject,
      })
      try {
        this.#write(payload)
      } catch (error) {
        this.pending.delete(id)
        reject(error)
      }
    })
    // 调用方不一定 await done（例如 fire-and-forget），避免未处理的 rejection
    done.catch(() => {})
    return { id, done, cancel: () => this.cancel(id).catch(() => {}) }
  }

  async cancel(jobId) {
    return this.#request({ cmd: 'cancel', job: jobId })
  }

  async dispose() {
    if (this.disposed) return
    this.disposed = true
    const child = this.child
    if (!child) return
    try {
      child.stdin.end()
    } catch {}
    await new Promise((resolve) => {
      const timer = setTimeout(() => {
        try {
          child.kill()
        } catch {}
        resolve(undefined)
      }, 3000)
      child.once('exit', () => {
        clearTimeout(timer)
        resolve(undefined)
      })
    })
    this.child = undefined
  }
}

/** 把交织 s16le PCM 拼成 WAV 文件内容（44 字节头 + 数据）。 */
export function pcmToWav(pcm, format = AUDIO_FORMAT) {
  const data = Buffer.concat(Array.isArray(pcm) ? pcm : [pcm])
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + data.length, 4)
  header.write('WAVE', 8)
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(format.channels, 22)
  header.writeUInt32LE(format.rate, 24)
  header.writeUInt32LE(format.rate * format.channels * 2, 28)
  header.writeUInt16LE(format.channels * 2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(data.length, 40)
  return Buffer.concat([header, data])
}

/** 一次性合成到 WAV 文件，返回 { file, job }。 */
export async function synthesizeToFile(tts, options) {
  const chunks = []
  const job = tts.speak({
    ...options,
    onAudio: (pcm) => chunks.push(pcm),
  })
  const result = await job.done
  const dir = options.outDir ?? path.join(os.tmpdir(), 'moss-tts')
  fs.mkdirSync(dir, { recursive: true })
  const file = path.join(dir, `${options.name ?? result.id}.wav`)
  fs.writeFileSync(file, pcmToWav(chunks))
  return { file, job: result }
}
