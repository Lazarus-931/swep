CXX     = xcrun -sdk macosx clang++
SYSROOT = $(shell xcrun --sdk macosx --show-sdk-path)
CXXFLAGS = -std=c++17 -O2 -isysroot $(SYSROOT) \
           -framework Metal -framework Foundation -fobjc-arc

all: bench

bench: main.mm
	$(CXX) $(CXXFLAGS) -o $@ $<

run: bench
	./bench

plot: results.csv
	python3 plot.py

clean:
	rm -f bench results.csv

.PHONY: all run plot clean
