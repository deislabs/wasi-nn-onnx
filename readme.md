# ONNX implementation for WASI-NN

This project is an ONNX implementation of the WASI-NN project, built by the
Bytecode Alliance.

This is experimental work in progress that is not guaranteed to compile or run.

### Notes

- [`crates/wasi-nn-onnx-wasmtime`](crates/wasi-nn-onnx-wasmtime) contains the
  Wasmtime implementation of the WASI-NN spec for ONNX. It is modeled and
  follows
  [the OpenVINO implementation from the Wasmtime repo](https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-nn).
- [`crates/wasi-nn`](crates/wasi-nn) contains a fork of the
  [guest bindings for WASN-NN](https://github.com/bytecodealliance/wasi-nn),
  modified to include ONNX options.
