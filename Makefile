CXX      = xcrun -sdk macosx clang++
SYSROOT  = $(shell xcrun --sdk macosx --show-sdk-path)

CXXFLAGS = -std=c++17 -O2 -isysroot $(SYSROOT) \
           -framework Metal -framework Foundation -framework QuartzCore \
           -fobjc-arc

TARGET   = GPUProbe

all: $(TARGET)

$(TARGET): main.mm
	$(CXX) $(CXXFLAGS) -o $@ $<

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
