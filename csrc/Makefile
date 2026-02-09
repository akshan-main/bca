# Makefile for CAG C++ acceleration core
#
# Build:   make
# Clean:   make clean
# Test:    make test

CXX := g++
CXXFLAGS := -O3 -std=c++17 -fPIC -Wall -Wextra -DNDEBUG
LDFLAGS := -shared

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    TARGET := cag_fast.dylib
    LDFLAGS += -dynamiclib -undefined dynamic_lookup
    # Use Apple Clang if available
    CXX := $(shell which clang++ 2>/dev/null || echo g++)
else ifeq ($(UNAME_S),Linux)
    TARGET := cag_fast.so
else
    TARGET := cag_fast.dll
endif

SRC := cag_fast.cpp
OBJ := $(SRC:.cpp=.o)

.PHONY: all clean test install

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Built $(TARGET) successfully"

clean:
	rm -f $(TARGET) $(OBJ) test_cag

test: $(TARGET)
	@echo "C++ library built and available at $(TARGET)"
	@echo "Run 'python -c \"from cegraph.context._native import NativeCAG; print(NativeCAG.is_available())\"' to verify"

# Install to the cegraph package directory
install: $(TARGET)
	cp $(TARGET) ../src/cegraph/context/
	@echo "Installed $(TARGET) to src/cegraph/context/"
