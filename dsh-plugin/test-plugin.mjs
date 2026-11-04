// 用假的 ctx 验证插件入口能加载、能注册工具、能真的合成。
// 不需要 DSH，也不会出声（say 用 save_to）。
//
//   node dsh-plugin/test-plugin.mjs
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { apply, name } from './index.mjs'

const exe = path.resolve(process.env.MOSS_TTS_EXE ?? path.join('csrc', 'build', 'moss_tts.exe'))
const models = path.resolve(process.env.MOSS_TTS_MODELS ?? 'models')

const tools = new Map()
const effects = []
const ctx = {
  tools: { register: (tool) => tools.set(tool.name, tool) },
  effect: (factory, label) => effects.push({ factory, label }),
  logger: { info: () => {}, warn: () => {} },
}

const failures = []
const check = (n, ok, detail = '') => {
  console.log(`${ok ? '[PASS]' : '[FAIL]'} ${n}${detail ? ` — ${detail}` : ''}`)
  if (!ok) failures.push(n)
}

try {
  await apply(ctx, { exe, models, voice: 'Junhao' })
  check('插件名', name === 'moss-tts')
  check('注册了 4 个工具', tools.size === 4, [...tools.keys()].join(','))
  check('工具齐全',
    ['moss_tts_say', 'moss_tts_voices', 'moss_tts_stop', 'moss_tts_clone'].every((t) => tools.has(t)))
  check('注册了卸载 effect', effects.length === 1 && typeof effects[0].factory() === 'function', effects[0]?.label)

  const voices = await tools.get('moss_tts_voices').execute({}, {})
  check('voices 工具可用', /内置（\d+）/.test(voices.detail), voices.detail.split('\n')[0].slice(0, 60))

  const outFile = path.resolve('runs', 'plugin_out', 'plugin-entry.wav')
  const said = await tools.get('moss_tts_say').execute(
    { text: 'plugin entry test.', voice: 'Junhao', save_to: outFile }, {})
  check('say 工具生成 WAV', fs.existsSync(outFile) && fs.statSync(outFile).size > 44,
    `${said.seconds.toFixed(2)} s, ${fs.existsSync(outFile) ? fs.statSync(outFile).size : 0} bytes`)
  check('say 返回统计', said.compute_ms > 0 && said.rtf > 0, `compute=${Math.round(said.compute_ms)}ms rtf=${said.rtf.toFixed(3)}`)

  const stop = await tools.get('moss_tts_stop').execute({}, {})
  check('stop 工具可用', typeof stop.detail === 'string', stop.detail)

  // 卸载：应把常驻子进程收掉
  await effects[0].factory()()
  check('卸载不抛错', true)
} catch (error) {
  check('未捕获异常', false, error?.stack ?? String(error))
}

console.log()
if (failures.length) {
  console.log(`${failures.length} 项失败：${failures.join('、')}`)
  process.exit(1)
}
console.log('全部通过')
