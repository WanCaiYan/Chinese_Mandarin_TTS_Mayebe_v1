
###########################################################
#
#	Input and output directories
#
###########################################################

PREFIX  := ..
BIN_DIR := $(PREFIX)/binary/Linux
OBJ_DIR := $(PREFIX)/binary/object/Linux

OUTDIRS :=
OUTDIRS += $(BIN_DIR)
OUTDIRS += $(addprefix $(OBJ_DIR)/,\
	demo)

INCLUDE := $(PREFIX)/engine


###########################################################
#
#	Source and object files
#
###########################################################

LIBS :=
LIBS += $(addprefix $(BIN_DIR)/,\
	libttsputonghua.a libttsbase.a libcsttools.a libhtslib.a)

SRCS :=
SRCS += $(addprefix demo/,\
	demo.cpp)

OBJS := $(addprefix $(OBJ_DIR)/, \
	$(patsubst %.cpp, %.o, $(filter %.cpp, $(SRCS))) )

# Compiling flags
CPPFLAGS += $(patsubst %, -I%, $(INCLUDE)) -DUNICODE -DCRYSTALDLL_EXPORTS -g


###########################################################
#
#	Target compiling and linking commands
#
###########################################################

BIN_NAME:= CrystalDllDemo

.PHONY: lib clean

lib: $(OUTDIRS) $(BIN_DIR)/$(BIN_NAME)

$(BIN_DIR)/$(BIN_NAME): $(OBJS)
	@echo Creating library $@
	$(CXX) -o $@ $^ /usr/local/lib/libiconv.so /usr/lib/CrystalDll.so
#	$(CXX) -o $@ $^ -liconv -L/usr/local/libiconv/lib

$(OBJS): $(OBJ_DIR)/%.o: $(addprefix $(PREFIX)/, %.cpp)
	@echo Compiling $<
	@$(CXX) $(CPPFLAGS) -c $< -o $@

# make output directories
# 1) will check the directory list
# 2) make directory if not exist
# 3) make parent directory if necessary (-p option)
# 4) dirs can also be retrieved from OBJS directly as
# 5) OBJDIRS := $(sort $(dir $(OBJS)))
$(OUTDIRS):
	@mkdir -p $@


###########################################################
#
#	Clean commands
#
###########################################################

clean:
	@echo Cleaning files
	@rm -f $(addsuffix /*.o,$(OUTDIRS))
	@rm -f $(addsuffix /*.d,$(OUTDIRS))
	@rm -f $(BIN_DIR)/$(BIN_NAME).*
