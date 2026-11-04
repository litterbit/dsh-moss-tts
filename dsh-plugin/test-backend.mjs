// 独立回归：不需要 DSH，直接用 node 跑（node >= 20）。
//
//   node dsh-plugin/test-backend.mjs [--exe csrc/build/moss_tts.exe] [--models models]
//
// 覆盖：启动 / info / voices / speak 流式字节数 / 取消 / clone / 落 WAV / dispose。
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { TtsBackend, PCM_BYTES_PER_MODEL_FRAME, pcmToWav, synthesizeToFile } from './backend.mjs'

const argv = process.argv.slice(2)
const pick = (flag, fallback) => {
  const index = argv.indexOf(flag)
  return index >= 0 ? argv[index + 1] : fallback
}
const exe = path.resolve(pick('--exe', path.join('csrc', 'build', 'moss_tts.exe')))
const models = path.resolve(pick('--models', 'models'))
const wav = path.resolve(pick('--wav', path.join('web', 'landing', 'assets', 'demo-clone.wav')))

const failures = []
const check = (name, ok, detail = '') => {
  console.log(`${ok ? '[PASS]' : '[FAIL]'} ${name}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures.push(name)
}

const tts = new TtsBackend({ exe, models, data: path.join('runs', 'plugin_userdata'), logger: (line) => console.log(line) })

try {
  const info = await tts.start()
  check('启动 + info', info?.info?.engine?.backend !== undefined, JSON.stringify(info?.info?.engine))

  const { builtin } = await tts.voices()
  check('voices', builtin.length > 0, `${builtin.length} 个内置`)

  // 1) 流式合成：字节数必须与帧数严格对应
  const chunks = []
  const job = tts.speak({ text: 'plugin backend test.', voice: 'Junhao', seed: 5, greedy: true, maxFrames: 24,
    onAudio: (pcm) => chunks.push(pcm) })
  const result = await job.done
  const pcm = Buffer.concat(chunks)
  check('speak 完成', result.state === 'completed', `state=${result.state} frames=${result.frames}`)
  check('PCM 字节数 = 帧数 × 15360', pcm.length === result.frames * PCM_BYTES_PER_MODEL_FRAME,
    `${pcm.length} bytes / ${result.frames} frames`)
  check('流式确实分块到达', chunks.length > 1, `${chunks.length} 块`)

  // 2) 落 WAV
  const outDir = path.join('runs', 'plugin_out')
  const { file } = await synthesizeToFile(tts, { text: 'file output test.', voice: 'Junhao', seed: 6, greedy: true,
    maxFrames: 16, outDir, name: 'file-test' })
  const stat = fs.statSync(file)
  check('WAV 文件', stat.size > 44 && fs.readFileSync(file).subarray(0, 4).toString() === 'RIFF',
    `${path.basename(file)} ${stat.size} bytes`)

  // 3) 取消（min_frames 保证取消落在运行中）
  const long = tts.speak({ text: 'cancellation test sentence.', voice: 'Junhao', seed: 0, greedy: true,
    minFrames: 200, maxFrames: 375, onAudio: () => {} })
  await new Promise((resolve) => setTimeout(resolve, 600))
  await long.cancel()
  const cancelled = await long.done
  check('取消', cancelled.state === 'cancelled', `state=${cancelled.state} frames=${cancelled.frames}`)

  // 4) 克隆
  if (fs.existsSync(wav)) {
    const cloned = await tts.clone('plugin_test', wav)
    check('clone', cloned?.type === 'cloned', String(cloned?.name))
    const after = await tts.voices()
    check('克隆音色出现在列表里', after.user.some((v) => v.id === 'plugin_test'),
      after.user.map((v) => v.id).join(','))
  } else {
    check('clone', false, `缺少 WAV：${wav}`)
  }

  // 5) PCM → WAV 头自洽
  const wavBytes = pcmToWav([Buffer.alloc(4)])
  check('pcmToWav 头', wavBytes.length === 48 && wavBytes.readUInt32LE(40) === 4)
} catch (error) {
  check('未捕获异常', false, error?.stack ?? String(error))
} finally {
  await tts.dispose()
  check('dispose 后进程退出', !tts.ready)
}

console.log()
if (failures.length) {
  console.log(`${failures.length} 项失败：${failures.join('、')}`)
  process.exit(1)
}
console.log('全部通过')
