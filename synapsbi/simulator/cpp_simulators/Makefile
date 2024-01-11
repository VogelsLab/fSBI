AURYNPATH = $(HOME)/auryn

# Path to Auryn include files
AURYNINC = $(AURYNPATH)/src

# Path to Auryn library
AURYNLIB = $(AURYNPATH)/build/release/src


# The following should not require updating in most cases
CXX = mpicxx
CXXFLAGS = -ansi -pipe -O3 -march=native -ffast-math -pedantic -I/usr/include -I$(AURYNINC)
# for IST cluster 
#LDFLAGS = $(AURYNLIB)/libauryn.a -L$(BOOST_LIBRARYDIR) -lboost_filesystem -lboost_system -lboost_program_options -lboost_mpi -lboost_serialization
# for workstations
LDFLAGS = $(AURYNLIB)/libauryn.a -lboost_filesystem -lboost_system -lboost_program_options -lboost_mpi -lboost_serialization

# OBJECTS
OBJECTS = SixParamConnection.o CVAIFGroup.o T4wvceciMLPConnection.o RandStimGroup.o RFConnection.o

# Add your simulation's file name here as default target
all:

sim_%: sim_%.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $< $(LDFLAGS) -o $(subst .o,,$<)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $<

.SECONDARY: $(OBJECTS)