# Repository Guidelines

## Project Structure & Module Organization

This is an Arduino library for Soldered Inkplate e-paper boards. Core library code lives in `src/`, with public entry points in `src/Inkplate.h` and `src/Inkplate.cpp`. Board-specific drivers, pins, and waveforms are under `src/boards/<BoardName>/`; reusable display, image, system, and peripheral support is under `src/graphics/`, `src/system/`, and `src/features/`. Example sketches are organized by board in `examples/<BoardName>/` and then by purpose, such as `Basic`, `Advanced`, `Diagnostics`, and `Projects`. Fonts and generated image assets are stored in `Fonts/`, `extras/`, and example-local asset folders.

## Build, Test, and Development Commands

Install the Inkplate board package before compiling locally. A typical smoke test is:

```sh
arduino-cli compile --fqbn soldered-inkplate-boards:esp32:Inkplate10 examples/Inkplate10/Basic/Inkplate10_Hello_World
```

Use the matching FQBN and example directory for the board you changed. GitHub Actions runs `arduino/compile-sketches` across the supported Inkplate boards and installs required libraries such as ArduinoJson, Time, AsyncTCP, and ESPAsyncWebServer.

Format changed C/C++ sources with clang-format 12:

```sh
clang-format -i src/Inkplate.cpp src/Inkplate.h
```

## Coding Style & Naming Conventions

Follow `.clang-format`: Microsoft base style, 4-space indentation, no tabs, preserved include order, and aligned consecutive macros. Keep public Arduino APIs in lower camel case, for example `drawPixel()` or `setRotation()`. Board classes and files use board-prefixed names such as `Inkplate10Driver`. New examples should follow the existing pattern: `examples/Inkplate10/Basic/Inkplate10_Feature_Name/Inkplate10_Feature_Name.ino`.

## Testing Guidelines

There is no standalone unit-test suite. Treat compile coverage of affected examples as the required validation. For changes in shared code under `src/graphics`, `src/system`, or `src/features`, compile at least one representative sketch for every impacted board family. For diagnostics or factory programming changes, compile the matching `Diagnostics` sketch and document any hardware-only checks performed.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit subjects such as `Fix Color images dithering algorithm`, plus occasional scoped Conventional Commit style like `feat(examples): ...` and `docs(examples): ...`. Keep subjects concise and mention the affected board or area when useful.

Pull requests should describe the change, list boards/examples compiled, link related issues, and include display photos or screenshots for visible rendering changes. Do not commit Wi-Fi credentials, API keys, generated build output, or local agent/tooling directories.
