MOSS-TTS CPU 推理引擎 {{TAG}}  （Windows x64）

一、三步开始
  1. 解压到任意目录，例如 D:\MossTTS
  2. 双击 moss_tts.exe（不带参数 = 启动本地服务并自动打开浏览器）
  3. 浏览器里输入文字，Ctrl+Enter 合成

  默认地址 http://127.0.0.1:8759/ ，只监听本机，不弹防火墙。
  首次运行 Windows 可能提示"已保护你的电脑"：更多信息 -> 仍要运行（未购买代码签名证书）。

<!--FULL-->  本包已自带权重（models/，约 160 MB），解压即用。
<!--LITE-->  本包不含权重：把 llm_q8x4.moss / decoder_f16.moss / encoder_f16.moss / tokenizer.mossvocab
<!--LITE-->  与 voices/ 放到 exe 同级的 models/ 目录下即可。

二、命令行
  moss_tts.exe --text "你好，世界。" --voice Junhao --out hello.wav
  moss_tts.exe --text "..." --pcm-raw - > out.pcm      （流式 s16le / 48 kHz / 立体声）
  moss_tts.exe --list-voices
  moss_clone.exe --encoder models/encoder_f16.moss --in ref.wav --out models/voices/me.mossclone --name me
  moss_tts.exe --version

三、程序自己的数据
  便携模式：models/ 在 exe 旁边时，缓存与克隆音色写到 exe 同级的 userdata/。
  共享模式：找不到 models/ 时，回退到 %LOCALAPPDATA%\MossTTS\。
  删掉 userdata/ 或 models/cache/ 都不会丢东西，只是下次首音慢一点（约 0.2 s）。

四、接入自己的程序
  本地 HTTP（只监听 127.0.0.1）：
    GET  /api/v1/health | /info | /voices
    POST /api/v1/tts/stream    JSON -> chunked s16le
    POST /api/v1/tts/wav       JSON -> 完整 WAV
    POST /api/v1/voices/clone?name=xxx
    GET  /api/v1/jobs/<id>     统计
    POST /api/v1/jobs/<id>/cancel
  C 接口：include/moss/tts.h + lib/moss_tts.lib（SDK 包里）
  插件通道：moss_tts.exe --stdio（JSON 行协议，见 doc/api.md）
  详见仓库 doc/api.md。

五、DSH 插件（DeepSeek Harness）
  本目录的 dsh-plugin\ 就是插件本体。在 profile 的 cordis.patch.yml 里加：
    - insert:
        - id: moss-tts
          name: <本目录>\dsh-plugin\index.mjs
          config:
            exe: <本目录>\moss_tts.exe
            models: <本目录>\models
            voice: Junhao
  不需要先起服务：插件自己 spawn 常驻子进程（--stdio）。

六、许可
  Apache-2.0，见 LICENSE；模型派生与改动说明见 NOTICE。
  请只使用你有权使用的声音做克隆。
