# swep

Utilizing the most out of our gpu starts with understanding it.

Swep makes a internal script that runs and reports everthing about your chip, albiet it m1 to m5 across thread size, memory and speed.


Note, for this to work, make sure you:

- have macOS 14+ with Command Line Tools installed (`xcode-select --install`)
- any Apple Silicon Mac (M1, M2...)


To get started, simply

1. clone
2. build
3. run


```bash
git clone https://github.com/Lazarus-931/swep.git && cd swep
make run
```

Results print to stdout and a report is saved to `runs/<chip>.md` based on your device.
