

# This Makefile assumes the directory containing the or-tools library is
# already compiled and available below:
OR_DIR = ../or-tools-6.0

# There should be nothing else to change below. Just type make.

BIN_DIR = bin
OBJ_DIR = objs
SRC_DIR = src

CCC = g++
CFLAGS = -fPIC -std=c++0x -fwrapv -O4 -DNDEBUG 
INCLUDE = -I. -I$(OR_DIR) -I$(OR_DIR)/ortools/gen -I$(OR_DIR)/dependencies/install/include -DARCH_K8 -Wno-deprecated -DUSE_CBC -DUSE_CLP -DUSE_GLOP -DUSE_BOP -I/include
LIB = -Wl,-rpath $(OR_DIR)/lib -L$(OR_DIR)/lib -lortools -lz -lrt -lpthread
MKDIR = mkdir

all: \
  $(BIN_DIR)/ac_papers \
  $(BIN_DIR)/reviewer_papers \
  $(BIN_DIR)/sac_to_ac

$(OBJ_DIR)/ac_papers.o: $(SRC_DIR)/ac_papers.cc 
	$(CCC) $(CFLAGS) $(INCLUDE) -c $(SRC_DIR)/ac_papers.cc -o $(OBJ_DIR)/ac_papers.o
$(OBJ_DIR)/reviewer_papers.o: $(SRC_DIR)/reviewer_papers.cc 
	$(CCC) $(CFLAGS) $(INCLUDE) -c $(SRC_DIR)/reviewer_papers.cc -o $(OBJ_DIR)/reviewer_papers.o
$(OBJ_DIR)/sac_to_ac.o: $(SRC_DIR)/sac_to_ac.cc 
	$(CCC) $(CFLAGS) $(INCLUDE) -c $(SRC_DIR)/sac_to_ac.cc -o $(OBJ_DIR)/sac_to_ac.o

#binaries
$(BIN_DIR)/ac_papers: $(OBJ_DIR)/ac_papers.o
	$(CCC) $(CFLAGS) $(INCLUDE) $(OBJ_DIR)/ac_papers.o $(LIB) -o $(BIN_DIR)/ac_papers
$(BIN_DIR)/reviewer_papers: $(OBJ_DIR)/reviewer_papers.o
	$(CCC) $(CFLAGS) $(INCLUDE) $(OBJ_DIR)/reviewer_papers.o $(LIB) -o $(BIN_DIR)/reviewer_papers
$(BIN_DIR)/sac_to_ac: $(OBJ_DIR)/sac_to_ac.o
	$(CCC) $(CFLAGS) $(INCLUDE) $(OBJ_DIR)/sac_to_ac.o $(LIB) -o $(BIN_DIR)/sac_to_ac

clean:
	\rm -f $(BIN_DIR)/* $(OBJ_DIR)/*.o

