# Start here: understand and try your CV-DAW

You do not need to understand C++ to follow the first exercise in this guide.
Start with the Mac exercise. Leave the STM32 instructions for later.

## 1. What are we trying to build?

The final goal is to make music by moving your body:

```text
You move
    ↓
A camera sees you
    ↓
Software decides what your movement means
    ↓
A musical message is sent: “play this note” or “change this filter”
    ↓
The audio engine creates the sound
    ↓
You hear it through speakers or headphones
```

For example, your left arm could play a melody and your right leg could play a
kick drum. A pinch gesture could make a recorded pattern repeat.

**The complete camera-to-music instrument is not connected and finished yet.**
The latest work focuses on the message connection and the sound engine.

## 2. What did I actually add?

Think of this as building the wiring and sound-producing part of an instrument.

**Module 4 is the message connection.** One program puts messages in a small
shared area of computer memory. Another program reads them. A message can say
“play note 60 with the left arm.” The area holds up to 256 waiting messages.

**Module 5 is the sound engine.** It reads those messages, creates notes and drum
sounds, applies effects, and can repeat a recorded pattern. On the STM32 board,
it sends the finished sound to the Linux audio device.

**The mock sender is a pretend person.** It sends fake movement messages so we
can test the connection without a camera or MediaPipe.

The new files are in the `realtime` folder. Your original Python experiments,
including `BrazoIzquierdo.py`, were preserved.

### What you can expect today

| Part | Current state |
| --- | --- |
| Python sending messages to the C++ engine | Tested on this Mac |
| Generating and processing audio in memory | Tested on this Mac |
| Basic notes, drums, effects, and repeating patterns | Implemented; covered by host tests |
| Playing this new engine through the Mac speakers | Not supported by this version |
| Playing through the STM32 Linux audio device | Implemented; still needs board testing |
| Automatically turning camera movements into messages for this new engine | Still needs integration |
| Using the STM32 NPU for pose and hand tracking | Still needs model/delegate integration |
| Proving less than 35 ms delay and no audio dropouts | Still needs measurements on the board |

The project also contains an earlier pipeline outside `realtime`. It uses a
different message format. **Do not combine its brain/camera commands with the
new engine yet.** For the exercises below, use only the commands in this guide.

## 3. A few words you will see

| Word | Plain meaning |
| --- | --- |
| Terminal | An app where you type commands instead of clicking buttons |
| Source code | The instructions written by a programmer, such as a `.cpp` file |
| Build or compile | Turn source code into a program the computer can run |
| CMake | The tool that prepares the build |
| Engine | The program that creates and processes sound |
| Shared memory / IPC | The connection that lets the two programs exchange messages |
| Mock | A fake input used for testing |
| Offline mode | The engine calculates audio but does not send it to speakers |
| Stem | One part of a song, such as its drums or vocals, saved separately |
| XRUN | An audio buffer problem that can cause a click, gap, or interruption |

You do not have to memorize these words.

## 4. Your first exercise: test everything on your Mac

**Goal:** see that the engine works and that Python can talk to it.
**Expected sound:** none. This exercise is silent.
**You need:** this project, Terminal, CMake, a C++ compiler, and Python 3.
The build tools were available in this workspace when I tested it.

### Step 1 — Open Terminal

On your Mac, press **Command + Space**, type **Terminal**, and press **Return**.

Copy the commands from each box into Terminal. Press Return after pasting.
Do not type the Markdown fence marks around the command boxes.

### Step 2 — Go to your project folder

```sh
cd /Users/HaeJin/DJ
```

`cd` means “change directory.” This puts Terminal inside your project.

Check the location:

```sh
pwd
```

You should see:

```text
/Users/HaeJin/DJ
```

### Step 3 — Prepare the build

```sh
cmake -S realtime -B realtime/build -DCV_DAW_WITH_RUBBERBAND=OFF
```

This says: “prepare the programs in `realtime`, and put the generated build files
in `realtime/build`.” Turning Rubber Band off avoids needing the optional song
stretching library for this first test. Notes and drums still work.

You should see messages ending with something like:

```text
Build files have been written to: /Users/HaeJin/DJ/realtime/build
```

### Step 4 — Build the programs

```sh
cmake --build realtime/build --parallel 2
```

This creates the runnable engine and the small connection library used by Python.
Wait until Terminal finishes and lets you type another command.

You should see lines containing:

```text
Built target cv_daw_ipc
Built target cv-daw-engine
Built target cv-daw-rt-tests
```

Their order may differ. A warning from `miniaudio.h` is not necessarily a failed
build. If you see `error:` or the build stops unsuccessfully, stop here and share
the full error message.

### Step 5 — Run the automatic tests

```sh
ctest --test-dir realtime/build --output-on-failure
```

These tests run the sound calculations and automatically start a short
Python-to-C++ connection test. You do not need a second Terminal for this step.

The important result is:

```text
100% tests passed, 0 tests failed out of 2
```

If only one test is listed, Python may not have been found when CMake prepared
the build. If a test fails, its error will appear in the output.

**When both tests pass, you have completed the first exercise.** That result
means the tested software pieces work together on your Mac. It does not prove
the camera connection, speaker output, or real-time performance on the STM32.

## 5. Optional exercise: run the two programs yourself

This shows you the two sides of the connection separately. It is also silent.
You will use two Terminal tabs at the same time.

### Terminal tab A — Start the engine

In the first tab:

```sh
cd /Users/HaeJin/DJ
./realtime/build/cv-daw-engine --offline 22500
```

The `./` means “run a program from this folder.”
The number `22500` makes this demonstration run for approximately 60 seconds.

You should see:

```text
offline-ready
```

Leave this tab running. It is waiting for messages. It is normal that it does
not give you another command prompt while it is running.

### Terminal tab B — Send fake movement messages

Press **Command + T** to open another Terminal tab. Then run:

```sh
cd /Users/HaeJin/DJ
python3 realtime/vision_mock_sender.py --library realtime/build/libcv_daw_ipc.dylib --seconds 10
```

Use **`.dylib` on your Mac**. The `.so` extension in the Linux instructions is
for the board, not for macOS.

Wait about ten seconds. A successful result starts with:

```text
IPC PASS: sent=..., consumed=..., audio_frames_advanced=..., zero event drops
```

The actual numbers vary. Here is what they mean:

| Output | Meaning |
| --- | --- |
| `sent` | How many fake musical messages Python sent |
| `consumed` | How many messages the C++ engine read |
| `audio_frames_advanced` | The engine's audio clock kept moving |
| `zero event drops` | No messages were lost because the queue filled |

An `XRUNs=0` line in offline mode does **not** prove that physical audio playback
has no dropouts. Offline mode is not using the audio device.

### Stop the exercise

Return to tab A and press **Control + C**. This stops the engine and lets it
remove the temporary shared-memory connection. The engine also stops by itself
after about a minute.

If the sender has not finished, press Control + C in tab B too.

## 6. Why can't I hear anything on my Mac?

The latest engine was specifically written for **ALSA**, the Linux audio system
used on your target board. Your Mac uses a different audio system.

The Mac's offline mode calculates the audio and discards it. It does not play
through speakers or save a WAV file. Passing the tests therefore produces no
music you can listen to.

Running the engine on your Mac without `--offline` will report:

```text
ALSA/RT playback requires Linux; use --offline for host IPC testing
```

That is expected. A Mac playback mode would be an additional feature.

## 7. Later: hear the engine on the STM32 board

**Do not type these commands into your Mac Terminal unless that Terminal is
connected to the board's Linux shell.**

Before this stage, the board needs OpenSTLinux installed and booting, a working
audio output, the project files copied onto it, build tools, and the required
real-time permissions. The detailed administrator steps are in
[realtime/README.md](realtime/README.md), under “Linux RT privileges” and “Kernel
isolation.” Those steps change board settings; they are not part of the Mac test.

The example below assumes someone has placed this project at `/opt/cvdaw` on
the board and configured those permissions. If that folder does not exist, the
project has not been installed at the location used in this example.

### Board terminal A — Build and start the engine

```sh
cd /opt/cvdaw
cmake -S realtime -B realtime/build-board -DCMAKE_BUILD_TYPE=Release -DCV_DAW_CORTEX_A35=ON -DCV_DAW_WITH_RUBBERBAND=OFF
cmake --build realtime/build-board --parallel 2
./realtime/build-board/cv-daw-engine
```

This builds a separate **board version**. Do not copy the Mac executable onto
the board and expect it to run.

The engine may stay quiet in the terminal while running. This is normal.
If it prints an error and exits, resolve that error before starting the sender.

### Board terminal B — Send test notes

Open a second connection to the board and run:

```sh
cd /opt/cvdaw
python3 realtime/vision_mock_sender.py --library realtime/build-board/libcv_daw_ipc.so --seconds 10
```

Run both programs as the same Linux user. With a compatible, correctly configured
audio device and output routing, these fake messages should produce notes and
drum sounds through the board's audio output. This has not yet been verified on
your hardware.

Press **Control + C** in board terminal A to stop the engine.

Start with these built-in sounds. Loading song stems, enabling Rubber Band, and
connecting the camera are later steps.

## 8. If something goes wrong

| What you see | What to do |
| --- | --- |
| `cmake: command not found` | CMake is missing from that terminal's environment. Share the message and say whether this is the Mac or the board. |
| `No such file or directory` when starting the engine | Confirm you ran `cd` and completed the build. |
| Python cannot load the connection library | On Mac use `.dylib`; on Linux use `.so`. Confirm the path matches your build folder. |
| `No module named multiprocessing` or missing Python | Check `python3 --version` and share the result; the sender uses Python's standard library. |
| Shared memory not found | Start the engine first. Wait for `offline-ready` on the Mac, then start the sender before the engine finishes. |
| `another producer already owns the SPSC queue` | Run only one sender. Stop both programs, then restart the engine first. A previous crash may need additional cleanup. |
| `shm_open` failure | Another engine may be running, permissions may block shared memory, or a previous crash may have left a connection behind. Share the exact error before deleting anything. |
| `mlockall`, `SCHED_FIFO`, or `audio start: Operation not permitted` on the board | The board's real-time permissions need configuration; see the advanced guide. |
| `ALSA device initialization failed` on the board | The requested Linux audio device is unavailable, busy, or incompatible. Share the error and the output of `aplay -l` if that command is installed. |
| `ALSA negotiated incompatible geometry` | The audio device did not accept the required rate, channel count, or buffer sizes. This needs board-specific investigation. |
| Tests report a shared-memory permission error | Run them in your normal local Terminal. If that also fails, share the complete test output. |
| No sound during the Mac exercise | Expected: both exercises use silent offline mode. |

When asking for help, copy the command you ran and everything printed after it.
Also say **Mac** or **STM32 board**. You do not need to diagnose the error yourself.

## 9. Which files should you read?

You can run the exercises without reading any C++ files.

If you want to explore later:

| File | What it is for |
| --- | --- |
| [realtime/vision_mock_sender.py](realtime/vision_mock_sender.py) | The fake movement controller; a useful starting point for Python |
| [realtime/ipc_bridge.hpp](realtime/ipc_bridge.hpp) | The message format and shared connection |
| [realtime/audio_engine.hpp](realtime/audio_engine.hpp) | How messages become notes and repeating patterns |
| [realtime/dsp_primitives.hpp](realtime/dsp_primitives.hpp) | The calculations used to create and shape sound |
| [realtime/audio_engine.cpp](realtime/audio_engine.cpp) | Program startup and the Linux audio connection |
| [realtime/README.md](realtime/README.md) | Detailed build, hardware setup, and engineering notes |

You do not need to edit `third_party/miniaudio.h`. It is an included audio library.

## 10. What should you do first?

Open Terminal and follow **Section 4: Your first exercise**.
The first useful result is seeing **both tests pass**. After that, the next
milestone is hearing the fake controller on the STM32 board, followed by wiring
real camera movements into this same connection.
