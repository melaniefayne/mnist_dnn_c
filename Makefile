CXX = g++
CXXFLAGS = -std=c++11 -Wall
TARGET = main
SRCS = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
