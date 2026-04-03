# swep

Utilizing the most out of our gpu starts with understanding it.

Swep makes a internal script that runs and reports everthing about your chip, albiet it m1 to m5 across thread size, memory and speed.


Note, for this to work, make sure you:

- have macOS 14+ with Xcode installed (needs `xcodebuild`)
- any Apple Silicon Mac (M1, M2...)


To get started, simply

a. clone
b. build
c. run


```bash
git clone <repo-url> && cd swep
xcodebuild -project GPUProbe.xcodeproj -scheme GPUProbe -configuration Release build
$(xcodebuild -project GPUProbe.xcodeproj -scheme GPUProbe -configuration Release -showBuildSettings 2>/dev/null | grep -m1 BUILT_PRODUCTS_DIR | awk '{print $3}')/GPUProbe
```

