MAKEFLAGS += -j$(nproc)

# Compiler
NVCC = nvcc
CXX = g++

# Directory structure
SRC_DIR := src
CUDA_SRC_DIR := $(SRC_DIR)/cuda # Used for finding sources, not in VPATH directly
INCLUDE_DIR := include
OBJ_DIR := obj
LIB_DIR := lib

# --- Find all include directories (recursive) and create flags ---
ALL_INCLUDE_DIRS := $(shell find $(INCLUDE_DIR) -type d)
INCLUDE_FLAGS := $(patsubst %,-I%,$(ALL_INCLUDE_DIRS))

# Create directories if they don't exist
$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(LIB_DIR))

# Build configuration - set to release by default, override with make DEBUG=1
DEBUG ?= 0

# OpenCV configuration
OPENCV_INCLUDE := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

# Base compiler flags - Use INCLUDE_FLAGS
NVCC_BASE_FLAGS := -arch=sm_61 $(OPENCV_INCLUDE) $(INCLUDE_FLAGS)
CXX_BASE_FLAGS := -march=native $(OPENCV_INCLUDE) $(INCLUDE_FLAGS)

# Debug/Release specific flags
ifeq ($(DEBUG), 1)
    NVCC_FLAGS := $(NVCC_BASE_FLAGS) \
                 -G \
                 -g \
                 -O0 \
                 --generate-line-info \
                 --device-debug \
                 -Xcompiler "-rdynamic $(INCLUDE_FLAGS)" \
                 -Xptxas -O0,-v
    CXX_FLAGS := $(CXX_BASE_FLAGS) -g -O0
    $(info Building in DEBUG mode)
else
    NVCC_FLAGS := $(NVCC_BASE_FLAGS) \
        -O3 \
        --use_fast_math \
        -Xptxas --allow-expensive-optimizations=true,-v \
        --restrict \
        --extra-device-vectorization \
        -lineinfo \
        -Xcompiler "-O3 -march=native -mtune=native -funroll-loops -ffast-math -fomit-frame-pointer -fopenmp -DNDEBUG $(INCLUDE_FLAGS)" \
        -Xptxas -dlcm=ca \
        -Xptxas -O3 \
        -Xptxas -warn-spills

    CXX_FLAGS := $(CXX_BASE_FLAGS) \
        -O3 \
        -march=native \
        -mtune=native \
        -funroll-loops \
        -ffast-math \
        -fomit-frame-pointer \
        -fopenmp \
        -DNDEBUG
    $(info Building in RELEASE mode)
endif

# CUDA Libraries and Include paths
CUDA_LIBS := -lcudart -lcublas -lcurand
CUDA_PATH := /opt/cuda
CUDA_INC := -I$(CUDA_PATH)/include
CUDA_LIB := -L$(CUDA_PATH)/lib64

# --- Source files (recursive find) ---
# Find ALL sources under src potentially needed for the executable
ALL_CUDA_SRC := $(shell find $(SRC_DIR) -name '*.cu')
ALL_CPP_SRC := $(shell find $(SRC_DIR) -name '*.cpp')
# Define the directories containing sources for the library
LIB_CORE_DIR := $(SRC_DIR)/core
LIB_CUDA_DIR := $(SRC_DIR)/cuda
# Find sources specifically for the library from core and cuda directories
LIB_CUDA_SRC := $(shell find $(LIB_CORE_DIR) $(LIB_CUDA_DIR) -name '*.cu')
LIB_CPP_SRC := $(shell find $(LIB_CORE_DIR) $(LIB_CUDA_DIR) -name '*.cpp')
# Main source file (assuming it's in the root and only for the executable)
MAIN_SRC := main.cpp

# --- Object files ---
# Map source file names to object file names in OBJ_DIR using notdir
# WARNING: Requires unique base filenames across all subdirectories!

# Objects specifically for the library
LIB_CUDA_OBJ := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(notdir $(LIB_CUDA_SRC)))
LIB_CPP_OBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(LIB_CPP_SRC)))
LIB_OBJ := $(LIB_CUDA_OBJ) $(LIB_CPP_OBJ)

# Objects for the executable (ALL objects from src + main)
# Note: If the executable only needs the library + main, adjust this.
# This assumes the executable might need objects from src/ that are NOT in the library.
ALL_CUDA_OBJ := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(notdir $(ALL_CUDA_SRC)))
ALL_CPP_OBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(ALL_CPP_SRC)))
MAIN_OBJ := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(MAIN_SRC)))
# Define ALL objects needed for the final executable link
ALL_OBJ := $(ALL_CUDA_OBJ) $(ALL_CPP_OBJ) $(MAIN_OBJ)

# VPATH tells make where to find prerequisites (source files)
# Creates a colon-separated list of all directories under SRC_DIR and the root for main.cpp
VPATH := $(shell find $(SRC_DIR) -type d | tr '\n' ':').

# Executable name
EXE := neuralnetwork
# Library name
LIB_NAME := cpplib_thes.a

# Default rule: build the executable
all: $(EXE)

# --- Executable Build ---
debug:
	$(MAKE) -j$(nproc) DEBUG=1 $(EXE)

release:
	$(MAKE) -j$(nproc) DEBUG=0 $(EXE)

# Rule to link the object files and create the final executable
# Depends on ALL object files needed for the executable
$(EXE): $(ALL_OBJ)
	@echo "Linking executable: $@"
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(CUDA_LIB) $(CUDA_LIBS) $(OPENCV_LIBS)

# --- Library Build ---
library: $(LIB_DIR)/$(LIB_NAME)

# Depends ONLY on library object files
$(LIB_DIR)/$(LIB_NAME): $(LIB_OBJ)
	@echo "Creating library: $@"
	ar rcs $@ $^

debug-lib:
	$(MAKE) -j$(nproc) DEBUG=1 library

release-lib:
	$(MAKE) -j$(nproc) DEBUG=0 library

# --- Common Compilation Rules ---
# These rules use VPATH to find the source files

# Rule for main.cpp (specific case, as it's not under src/)
$(OBJ_DIR)/main.o: $(MAIN_SRC)
	@echo "Compiling main: $< -> $@"
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c -o $@ $<

# Rule for CUDA files (using VPATH)
# Creates obj/file.o from src/path/to/file.cu
$(OBJ_DIR)/%.o: %.cu
	@echo "Compiling CUDA: $< -> $@"
	$(NVCC) $(NVCC_FLAGS) $(CUDA_INC) -c -o $@ $<

# Rule for C++ files (using VPATH)
# Creates obj/file.o from src/path/to/file.cpp
$(OBJ_DIR)/%.o: %.cpp
	@echo "Compiling C++: $< -> $@"
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c -o $@ $<

# Clean up
clean:
	@echo "Cleaning object files, executable, and library..."
	rm -f $(OBJ_DIR)/*.o $(EXE) $(LIB_DIR)/$(LIB_NAME)

.PHONY: all debug release library debug-lib release-lib clean