METAL    = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
CXX      = xcrun -sdk macosx clang++
SYSROOT  = $(shell xcrun --sdk macosx --show-sdk-path)

CXXFLAGS = -std=c++17 -O2 -isysroot $(SYSROOT) \
           -framework Metal -framework Foundation -framework QuartzCore \
           -fobjc-arc

TARGET   = GPUProbe
SHADER   = swep_.metal
METALOBJ = swep_.air
METALBIN = swep_.metallib

all: $(TARGET) $(METALBIN)

$(TARGET): main.mm
	$(CXX) $(CXXFLAGS) -o $@ $<

$(METALBIN): $(SHADER)
	$(METAL) -c -o $(METALOBJ) $<
	$(METALLIB) -o $@ $(METALOBJ)
	rm -f $(METALOBJ)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) $(METALOBJ) $(METALBIN)

.PHONY: all run clean
