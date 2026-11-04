// DSH 宿主插件入口：把 MOSS-TTS 引擎接成可调用的语音工具。
//
// 加载方式（profile 的 cordis.patch.yml）：
//   - insert:
//       - id: moss-tts
//         name: D:\path\to\dsh-moss-tts\dsh-plugin\index.mjs
//         config:
//           exe: D:\MossTTS\moss_tts.exe        # 可省，默认 %LOCALAPPDATA%\MossTTS 或仓库 dist
//           models: D:\MossTTS\models           # 可省
//           voice: Junhao
//
// 设计：一个常驻子进程（--stdio），权重只加载一次；工具调用是薄的。
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import { TtsBackend, pcmToWav, resolveModels, defaultExe } from './backend.mjs'
import { playWavFile } from './player.mjs'

export const name = 'moss-tts'
export const inject = ['tools']

const TOOL_DIRS = [
  process.resourcesPath && path.join(process.resourcesPath, 'app.asar', 'dsh', 'node_modules', '@deepseek-ai'),
  'D:/7software/AI/dsh/resources/app.asar/dsh/node_modules/@deepseek-ai',
].filter(Boolean)

async function loadDefineTool() {
  for (const dir of TOOL_DIRS) {
    try {
      const module = await import(pathToFileURL(path.join(dir, 'dsh-tools', 'lib', 'index.js')).href)
      if (typeof module.defineTool === 'function') return module.defineTool
    } catch {
      // 换下一个候选路径
    }
  }
  // 独立运行/宿主升级后的兜底：工具定义本身就是普通对象
  return (spec) => spec
}

export async function apply(ctx, config = {}) {
  const logger = (line) => {
    try {
      ctx.logger?.info?.(line)
    } catch {
      // 宿主没提供 logger 就静默
    }
  }

  const exe = config.exe ?? defaultExe()
  const models = config.models ?? (exe ? resolveModels(exe) : undefined)
  const tts = new TtsBackend({
    exe,
    models,
    data: config.data,
    llmThreads: config.llmThreads ?? 0,
    decodeThreads: config.decodeThreads ?? 0,
    logger,
  })
  const defaultVoice = config.voice ?? 'Junhao'

  let current = { job: undefined, abort: new AbortController() }

  const ensureStarted = async () => {
    if (!tts.ready) await tts.start()
  }

  ctx.effect(
    () => async () => {
      try {
        current.abort.abort()
        current.job?.cancel()
      } catch {}
      await tts.dispose()
    },
    'moss-tts.runtime',
  )

  const defineTool = await loadDefineTool()

  ctx.tools.register(
    defineTool({
      name: 'moss_tts_say',
      description:
        '用本机 MOSS-TTS 引擎把一段文字读出来（离线、CPU、不联网）。会阻塞到播放结束，适合"把结果念给用户听"。',
      parameters: {
        text: { type: 'string', required: true, description: '要朗读的文字。' },
        voice: { type: 'string', required: false, description: '内置音色 id（默认取插件配置）。' },
        voice_scope: { type: 'string', required: false, description: 'builtin（默认）或 user（自己克隆的音色）。' },
        wait: { type: 'boolean', required: false, description: '是否等播完再返回，默认 true。' },
        save_to: { type: 'string', required: false, description: '给了就把 WAV 存到这个路径（不播放）。' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            voice: { type: 'string', required: true },
            seconds: { type: 'number', required: true },
            compute_ms: { type: 'number', required: true },
            rtf: { type: 'number', required: true },
            file: { type: 'string', required: true },
          },
        },
        render: (_args, value) =>
          [{ type: 'text', text: `🔊 ${value.voice} · ${value.seconds.toFixed(2)} s · 计算 ${Math.round(value.compute_ms)} ms · RTF ${value.rtf.toFixed(3)}` }],
      },
      async execute(args) {
        const text = String(args.text ?? '').trim()
        if (!text) throw new Error('text 不能为空')
        await ensureStarted()

        const chunks = []
        const job = tts.speak({
          text,
          voice: String(args.voice ?? defaultVoice),
          scope: String(args.voice_scope ?? 'builtin'),
          onAudio: (pcm) => chunks.push(pcm),
        })
        current.job = job
        const info = await job.done
        current.job = undefined
        if (info.state === 'failed') throw new Error(info.error || '合成失败')

        const pcm = Buffer.concat(chunks)
        const outDir = config.data ?? path.join(process.env.LOCALAPPDATA ?? process.cwd(), 'MossTTS', 'cache')
        fs.mkdirSync(outDir, { recursive: true })
        const file = path.resolve(String(args.save_to ?? path.join(outDir, `say-${info.id}.wav`)))
        fs.writeFileSync(file, pcmToWav(pcm))

        if (!args.save_to && args.wait !== false) {
          current.abort = new AbortController()
          await playWavFile(file, { signal: current.abort.signal })
        }
        return {
          voice: String(args.voice ?? defaultVoice),
          seconds: info.audio_ms / 1000,
          compute_ms: info.compute_ms,
          rtf: info.rtf,
          file,
        }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'moss_tts_voices',
      description: '列出本机可用的 MOSS-TTS 音色（内置 + 自己克隆的）。',
      parameters: {},
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: { detail: { type: 'string', required: true } },
        },
        render: (_args, value) => [{ type: 'text', text: value.detail }],
      },
      async execute() {
        await ensureStarted()
        const { builtin, user } = await tts.voices()
        const lines = []
        lines.push(`内置（${builtin.length}）：${builtin.map((v) => v.id).join('、')}`)
        if (user.length) lines.push(`我的（${user.length}）：${user.map((v) => `${v.id}(${v.seconds.toFixed(1)}s)`).join('、')}`)
        return { detail: lines.join('\n') }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'moss_tts_stop',
      description: '立刻停止当前正在合成/播放的语音。',
      parameters: {},
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: { detail: { type: 'string', required: true } },
        },
        render: (_args, value) => [{ type: 'text', text: value.detail }],
      },
      async execute() {
        const job = current.job
        current.abort.abort()
        current.abort = new AbortController()
        if (!job) return { detail: '当前没有正在播放的语音。' }
        job.cancel()
        return { detail: '已请求停止。' }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'moss_tts_clone',
      description: '用一段 5~15 秒的 WAV 参考音频创建克隆音色，存到本机用户音色目录。',
      parameters: {
        name: { type: 'string', required: true, description: '音色名（字母/数字/_/-，≤32）。' },
        wav_path: { type: 'string', required: true, description: '参考音频的 WAV 路径（PCM；压缩格式先转）。' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: { detail: { type: 'string', required: true } },
        },
        render: (_args, value) => [{ type: 'text', text: value.detail }],
      },
      async execute(args) {
        await ensureStarted()
        await tts.clone(String(args.name), String(args.wav_path))
        return { detail: `音色 ${args.name} 已创建，之后用 voice="${args.name}" + voice_scope="user" 使用。` }
      },
    }),
  )

  logger(`moss-tts plugin ready: exe=${exe ?? '?'} models=${models ?? '?'}`)
}
