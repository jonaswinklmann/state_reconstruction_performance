CC=c++
INCLUDE=-I../extern/pybind11/include -Iinclude -I../extern/eigen $(shell python3-config --includes)
CXXFLAGS=-O3 -Wall -shared -std=c++17 -fPIC $(INCLUDE)
PLATFORM:=$(word 2,$(MAKECMDGOALS))

TARGET_DIR=../state_reconstruction/
SRC_DIR	:= src
OBJ_DIR	:= obj
BIN_DIR	:= bin

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES := $(wildcard $(SRC_DIR)/*.cu)
OBJECTS	:= $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
ifeq ($(PLATFORM),gpu)
CUDA_OBJECTS := $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CXXFLAGS += -lcudart -L/usr/local/cuda/lib64 -DCUDA
endif

all: module $(PLATFORM)
fresh:
	-rm $(OBJECTS)
	-rm $(CUDA_OBJECTS)
	-rm $(BIN_DIR)/state_reconstruction_cpp$(shell python3-config --extension-suffix)
	make module $(PLATFORM)

module: 
	make $(BIN_DIR)/state_reconstruction_cpp $(PLATFORM)
$(BIN_DIR)/state_reconstruction_cpp: $(OBJECTS) $(CUDA_OBJECTS) | $(BIN_DIR) $(TARGET_DIR)
	$(CC) $^ -o $@$(shell python3-config --extension-suffix) $(CXXFLAGS)
	cp $@$(shell python3-config --extension-suffix) $(TARGET_DIR)$(notdir $@$(shell python3-config --extension-suffix))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) -c $< -o $@ $(CXXFLAGS)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
ifeq ($(PLATFORM),gpu)
	nvcc -ccbin $(CC) -c $< -o $@ -Xcompiler "$(CXXFLAGS)"
endif
$(BIN_DIR) $(OBJ_DIR) $(TARGET_DIR):
	mkdir -p $@