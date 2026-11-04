// Windows 上的极简播放器：合成完的 WAV 交给系统 SoundPlayer。
//
// 为什么不用 Web Audio / 流式播放：
//   - 插件跑在 Node 侧，没有 AudioContext；
//   - 引 ffplay/sox 会给用户加依赖。
//   合成一段 5 秒语音约 0.7 秒，先落盘再播的延迟完全可接受。
//   需要毫秒级首音时再用「边收边播」，那是 v2 的事。
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

function psLiteral(value) {
  return String(value).replace(/'/g, "''")
}

/** 播放一个 WAV 文件，等它播完。返回 { code, canceled }。 */
export function playWavFile(file, options = {}) {
  const script = [
    '$ErrorActionPreference = "Stop"',
    `$player = New-Object System.Media.SoundPlayer '${psLiteral(file)}'`,
    '$player.PlaySync()',
  ].join('; ')

  return new Promise((resolve) => {
    const child = spawn('powershell', ['-NoProfile', '-NonInteractive', '-Command', script], {
      windowsHide: true,
      stdio: 'ignore',
    })
    if (options.signal) {
      options.signal.addEventListener('abort', () => {
        try {
          child.kill()
        } catch {}
      }, { once: true })
    }
    child.on('exit', (code) => resolve({ code, canceled: options.signal?.aborted === true }))
    child.on('error', () => resolve({ code: -1, canceled: false }))
  })
}

/** 写临时 WAV 并播放；播完删文件。 */
export async function playPcm(pcm, format, options = {}) {
  const dir = options.dir ?? path.join(os.tmpdir(), 'moss-tts')
  fs.mkdirSync(dir, { recursive: true })
  const file = path.join(dir, `say-${Date.now()}-${Math.random().toString(16).slice(2, 8)}.wav`)
  const { pcmToWav } = await import('./backend.mjs')
  fs.writeFileSync(file, pcmToWav(pcm, format))
  try {
    return await playWavFile(file, options)
  } finally {
    try {
      fs.unlinkSync(file)
    } catch {}
  }
}
