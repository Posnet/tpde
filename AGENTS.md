TPDE Project Overview
=====================

This repository hosts **TPDE**, a fast compiler back-end framework. The C++
implementation targets ELF based x86-64 and AArch64 systems and aims for low
latency compilation comparable to a baseline JIT.  Major components are:

* **tpde** – core framework handling register allocation, stack management,
  unwind info and object file emission.
* **tpde-encodegen** – generates encoding helpers from high level C code.
* **tpde-llvm** – LLVM IR front-end built on TPDE.

The code heavily uses C++ templates and concepts for static polymorphism.
Documentation inside `docs/tpde` describes the compiler architecture, the
analyzer pass, register allocation and instruction selection API.

Rust Port Goals
---------------

A Rust rewrite is in its infancy in `rust/` with minimal placeholder crates:
`tpde`, `tpde-llvm` and `tpde-encodegen`.  The intent is to replicate TPDE's
functionality using Rust idioms.  We rely on the `inkwell` crate for LLVM API
access and favor traits over class-style inheritance.

Coding Style
------------

* **Pragmatic** and direct code—keep dependencies light and code easy to reason
  about. Channel the mindset of Jonathan Blow, Casey Muratori and John Carmack:
  performance aware, straightforward, minimal abstractions.
* Use traits for polymorphism and keep generic usage simple.
* Prefer explicit, short functions. Avoid deep module hierarchies.
* Document modules and functions concisely in doc comments.

Operational Notes
-----------------

This Codex agent runs in a network isolated environment. `cargo` commands must be
invoked with `--offline`. Dependencies are expected to be vendored under
`rust/vendor`. If they are missing the agent should commit any current work and
report the issue.

