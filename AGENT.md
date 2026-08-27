# Project instructions

## Project
This repository is developed locally but compiled and tested on a remote
Linux server with Ascend/CANN hardware.

## Editing rules
- Keep changes minimal and scoped to the requested task.
- Do not refactor unrelated code.
- Follow the existing code style and naming conventions.
- Do not modify generated files or build artifacts.
- Do not commit binaries, logs, core dumps, or build directories.

## Remote-only environment
The local machine does not have the Ascend/CANN runtime or NPU hardware.

Therefore:
- Do not assume CANN/NPU tests can run locally.
- Perform static checks locally when possible.
- Do not change code merely because remote-only dependencies are unavailable.
- After editing, tell me exactly which commands should be run on the remote server.

## Git
- Show the changed files before finishing.
- Summarize the purpose of each change.
- Do not push automatically unless explicitly requested.

## Testing
Remote validation is performed on Jupiter.
Typical workflow:

git pull
<build command>
<test command>